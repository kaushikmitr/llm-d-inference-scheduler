# P2P Source Producer Plugin

**Type:** `p2p-source-producer`

Selects a peer that can supply the most cached prompt prefix and sets the
`x-kv-cache-source-host-port` header so the routing sidecar can pull that
prefix instead of recomputing it. Source selection runs in the `DataProducer`
phase before scheduling; the header is emitted in `PreRequest` after the
computing endpoint is known.

The plugin consumes per-endpoint `PrefixCacheMatchInfo` from an approximate or
precise prefix-cache producer. It samples among endpoints within one block of
the largest eligible prefix, weighted by `1/(1+waiting queue)` and a
request-ID hash. This spreads pulls across replicas that hold nearly identical
prefixes. After scheduling, the header is set only when the sampled source
out-caches the computing endpoint by at least `minCachedTokenDelta` tokens.
Any inbound value of the header is removed.

## Reusable Prefix Floor

Per-tier cache data distinguishes the source's pull-servable CPU blocks from
its GPU-only blocks. For a sampled source with a confirmed contiguous CPU
prefix, the plugin publishes a name-bound `ReusablePrefixTokens` floor for
scheduling consumers. Let `S` be that prefix length and `D` be
`minCachedTokenDelta`:

```text
G = max(0, S - D + 1)
```

The attribute is omitted when `G` is zero.

The [`ReusablePrefixTokens` type](../../../datalayer/attribute/prefix/data_types.go)
defines the local-or-pull invariant that scheduling consumers rely on.

Tierless approximate match data remains eligible for header selection because
it can still identify a likely source. It does not confirm a CPU-tier prefix,
so it does not publish `ReusablePrefixTokens` for scheduling. A
[`context-length-aware`](../../../scheduling/scorer/contextlengthaware/README.md)
consumer leaves the total prompt length unchanged when no confirmed floor is
available.

## Parameters

- `prefixMatchInfoProducerName` (string, optional): Name of the prefix-cache producer instance that supplies `PrefixCacheMatchInfo`. Empty selects the default unnamed producer.
- `minCachedTokenDelta` (int, optional, default: `1`): Minimum cached-token advantage required to emit the source header. Must be `>= 1`. Higher values avoid transfers for prefixes that are cheap to recompute.
- `prefillProfileName` (string, optional, default: `prefill`): P/D disaggregation prefill profile containing the endpoint that computes the prefix. If the profile has no target, the primary profile target is used.
- `costModel` (object, optional): When set, the header is emitted only when the pull's gain exceeds its cost, evaluated after the `minCachedTokenDelta` floor. Omitted, the fixed-delta decision applies. See [Cost model](#cost-model).

## Cost model

A pull saves the computing pod the prefill of `delta` tokens (the source's cached-token advantage) and costs the transfer of those tokens plus two waits that only apply on busy pods: a busy source is slow to start sending the blocks, and a busy computing pod re-admits the pulled request through its waiting queue after the blocks arrive. Both waits are constants of the busy state, not of queue depth. The header is emitted when

```text
delta * (prefill - transfer) + fleetWeight * busy(d) * delta * prefill
  > transferFixed + busy(s) * sourceWait + busy(d) * requeue
```

where `busy(x)` is true when pod `x` has at least `busyQueueThreshold` requests waiting. `fleetWeight` credits a pull on a busy computing pod with the prefill time it frees for the other requests there; at 0 the decision optimizes the pulled request's own latency only.

The cost model also changes how the source is chosen. Instead of sampling within one block of the largest prefix weighted by queue depth, every source is ranked by the wait it adds (`sourceWait` if busy) plus the recompute the computing pod pays for the tokens it holds short of the best-cached source, `(maxCached - cached) * (prefill - transfer)`. Sources within one block's recompute of the minimum are sampled uniformly by request-ID hash. An idle source therefore beats a busy one unless the busy one's extra cache is worth more than the source wait; with the example values below, a busy source needs about 23K more cached tokens than an idle one to be chosen over it.

Parameters, all measured on the serving fleet (an idle ladder of pull versus recompute at several prefix lengths gives the first three; one probe each with a busy source and a busy computing pod gives the waits):

- `prefillMicrosecondsPerToken` (float, required): prefill cost per token on the computing pod.
- `transferMicrosecondsPerToken` (float, required): P2P transfer cost per token; must be below the prefill cost.
- `transferFixedMs` (float, optional, default `0`): fixed cost of a pull on idle pods.
- `sourceWaitMs` (float, optional, default `0`): added when the source is busy.
- `requeueMs` (float, optional, default `0`): added when the computing pod is busy.
- `fleetWeight` (float, optional, default `0`): weight of the freed prefill time on a busy computing pod.
- `busyQueueThreshold` (int, optional, default `1`): waiting-queue depth at or above which a pod counts as busy.

Example values from a B200 fleet running gpt-oss-120b over RDMA: prefill 19, transfer 3.9, transferFixed 15, sourceWait 350, requeue 500. With those, the pull is emitted from about 1K tokens when both pods are idle and from about 34K tokens when the computing pod is busy.

```yaml
  - type: p2p-source-producer
    name: p2p-cache-source
    parameters:
      prefixMatchInfoProducerName: precise-cache
      minCachedTokenDelta: 256
      costModel:
        prefillMicrosecondsPerToken: 19
        transferMicrosecondsPerToken: 3.9
        transferFixedMs: 15
        sourceWaitMs: 350
        requeueMs: 500
        fleetWeight: 0
```

## Configuration

Cache-aware scheduling requires confirmed per-tier data, so use a named
`precise-prefix-cache-producer`. Confirmed local-cache accounting excludes
speculative entries when `speculativeIndexing` is enabled. The token producer
must render tokens exactly as the serving model does. This example disables
speculative indexing and shows the producer-side configuration; the consuming
work classifier is documented under [P2P cache-aware prefill work](../../../scheduling/scorer/contextlengthaware/README.md#p2p-cache-aware-prefill-work).

```yaml
plugins:
  - type: token-producer
    parameters:
      modelName: model-name
      vllm:
        url: http://render:8000
  - type: endpoint-notification-source
  - type: precise-prefix-cache-producer
    name: precise-cache
    parameters:
      tokenProcessorConfig:
        blockSizeTokens: 64
      kvEventsConfig:
        topicFilter: "kv@"
        discoverPods: true
        podDiscoveryConfig:
          socketPort: 5557
      speculativeIndexing: false
  - type: p2p-source-producer
    name: p2p-cache-source
    parameters:
      prefixMatchInfoProducerName: precise-cache
      prefillProfileName: prefill
      minCachedTokenDelta: 1
dataLayer:
  sources:
    - pluginRef: endpoint-notification-source
      extractors:
        - pluginRef: precise-cache
```

## Deployment Requirements

The source header results in a transfer only when the serving deployment can
serve and pull the named blocks:

- vLLM includes `OffloadingConnector` with a `p2p` secondary tier, and the routing sidecar is configured to consume the source header.
- Every potential source uses `offload_prompt_only: false`. The default `true` omits decode-phase blocks from its offload tier.
- Peers use identical `--block-size` values. vLLM rejects a mismatch with `block_len mismatch`.
- Peers use the same `PYTHONHASHSEED`, so their block hashes agree.
- The precise producer receives KV events from every endpoint that may serve as a source and reports per-tier cache data.

For cache-aware prefill-work routing, follow the
[`context-length-aware` profile and work-range guidance](../../../scheduling/scorer/contextlengthaware/README.md#p2p-cache-aware-prefill-work).

## Related Documentation

- [Context Length Aware Scorer](../../../scheduling/scorer/contextlengthaware/README.md)
- [Approximate Prefix Cache Producer](../approximateprefix/README.md)
- [Precise Prefix Cache Producer](../preciseprefixcache/README.md)
