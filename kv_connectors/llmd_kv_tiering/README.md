# KV tiering (`kvtierd`)

A node-level KV cache tier for LLM inference engines. A Go daemon, `kvtierd`, runs once per node and owns a shared-memory pool of KV blocks. Each engine loads a thin shim that copies blocks between the engine's own CPU offload tier and that pool.

Status: the daemon, the vLLM tier on GPU, and router hints with peer transfer, tested on H200 ([results](docs/results-2026-11-18-h200.md)). A shim for vLLM on TPU follows in a later change. See the [roadmap](docs/design.md#roadmap).

## How it relates to `llmd_fs_backend`

[`llmd_fs_backend`](https://github.com/llm-d/llm-d-kv-cache/tree/main/kv_connectors/llmd_fs_backend) in llm-d-kv-cache replaces vLLM's offloading spec and stores blocks as files on a shared filesystem. This tier plugs in differently: it is a secondary tier behind vLLM's own CPU tier, and it keeps blocks in host memory owned by a per-node daemon. The two can be used together. A future SSD tier under the daemon's pool could reuse the FS backend's file and GPUDirect Storage code.

## Why a daemon

- **Blocks outlive the engine.** The daemon owns the memory. When an engine restarts, the blocks it stored are still loadable.
- **Engines on a node share one pool.** A block one engine stored is a hit for every other engine with the same model layout.
- **One implementation for every engine and accelerator.** Policy, eviction, hints and peer transfer live in the daemon. Only the copy into the engine's memory is engine-specific.

## How it plugs into vLLM

vLLM's `TieringOffloadingSpec` loads out-of-tree secondary tiers through `module_path`. No vLLM patch is needed.

```bash
pip install ./kv_connectors/llmd_kv_tiering

vllm serve <model> \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "TieringOffloadingSpec",
      "cpu_bytes_to_use": 20000000000,
      "secondary_tiers": [{
        "type": "LLMDKVTierManager",
        "module_path": "llmd_kv_tiering.vllm_tier",
        "socket_path": "/var/run/llmd-kv-tiering/kvtierd.sock"
      }]
    }
  }'
```

Set `PYTHONHASHSEED` to the same value on every engine. vLLM's block hashes depend on it. Without it, blocks from before a restart, or from another engine, never match.

Tier options:

| Option | Default | Meaning |
| --- | --- | --- |
| `socket_path` | `/var/run/llmd-kv-tiering/kvtierd.sock` | The daemon's socket |
| `namespace` | derived from the model, KV layout, parallelism and vLLM version | Engines share blocks only within a namespace |
| `namespace_salt` | empty | Mixed into the derived namespace. Set it to the model revision, or anything else that changes the KV bytes |
| `num_io_threads` | 4 | Threads that copy blocks |
| `timeout_s` | 5 | Per-request timeout to the daemon |
| `reconnect_interval_s` | 1 | How often to retry a lost daemon |
| `locality` | unset | `LOCAL` or `REMOTE`, reported on KV events |
| `enable_kv_events` | true | Report this tier's blocks on vLLM's KV events, if KV events are enabled |

If the daemon is down, the tier misses every lookup and fails every store. vLLM recomputes. The tier reconnects on its own.

## Router hints

A router can steer the tier with four hint actions: `kv.fetch`, `kv.evict_on_fetch`, `kv.prefetch` and `kv.move`. They use vLLM's KV hint envelope, so routers that already emit these actions work unchanged. A router can attach them in either of two ways. Neither needs a vLLM change.

- In `kv_transfer_params["kv_hints"]` on an OpenAI API request. This is the path for llm-d's router.
- As vLLM's `kv_hints` argument, for callers that drive the engine in process, such as Dynamo.

```json
{"kv_transfer_params": {"kv_hints": {
  "protocol_version": "0.1",
  "message_id": "m-1",
  "actions": [{
    "action_id": "a-1",
    "action_type": "kv.fetch",
    "action_version": "1.0",
    "payload": {
      "source_control_endpoint": "tcp://10.0.0.5:9401",
      "block_hashes": [1234567890123456789]
    }
  }]
}}}
```

An endpoint names a node's daemon. `block_hashes` are the 64-bit block hashes the router already sees in vLLM's KV events. With a fetch hint, a lookup of a covered block reports a hit, and the load pulls the block from the source daemon over TCP. With `kv.prefetch` or `kv.move` sent to the source, the source offers the blocks and the destination pulls them.

How the tier reads fetch hints:

- **Versions must match.** An envelope whose `protocol_version` is not `0.1`, or an action whose `action_version` is not `1.0`, is rejected. A later version may change what the fields mean.
- **The first hint wins.** When several fetch hints name the same hash, the first one in the envelope is used.
- **A plain `kv.fetch` covers the whole request.** Any block of the request that is not local is fetched from the hint's source, not only the listed hashes. Routers rely on this because 64-bit hashes do not survive every hint path: vLLM's gRPC path carries numbers as doubles. So llm-d's router sends a placeholder hash. Blocks the source does not have are recomputed. `kv.evict_on_fetch` is limited to the hashes it lists, because it deletes the source copy. Set the tier option `opportunistic_query: false` to fetch only listed hashes.

## Running the daemon

```bash
go build -o bin/kvtierd ./cmd/kvtierd
./bin/kvtierd --socket /var/run/llmd-kv-tiering/kvtierd.sock --capacity 512GiB
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--capacity` | 16GiB | Pool size |
| `--slot-bytes` | 0 | Block size. 0 takes it from the first engine |
| `--hugetlb` | false | Back the pool with 2 MiB huge pages |
| `--prefault` | true | Touch the pool at start, so its memory is charged to the daemon's container |
| `--policy` | lru | Eviction policy |
| `--metrics-addr` | :9400 | `/metrics` and `/healthz` |
| `--release-grace` | 10s | How long a disconnected engine's pins and half-written slots are held before reuse. The engine may still be copying when its connection drops |
| `--peer-listen` | :9401 | TCP address for other daemons. Empty disables peer transfer |
| `--peer-endpoint` | the listen address | The endpoint routers put in hints, e.g. `tcp://10.0.0.5:9401` |
| `--peer-timeout` | 30s | Time limit for one peer transfer |
| `--peer-streams` | 8 | TCP connections one fetch spreads its blocks over |

On Kubernetes, run the daemon as a sidecar or a DaemonSet. Share the socket directory with the engine through an `emptyDir` or `hostPath` volume.

## Development

From the repository root:

```bash
go test -race ./pkg/kvtier/...
cd kv_connectors/llmd_kv_tiering && PYTHONPATH=<vllm source tree>:. python -m pytest tests
```

The Python tests build `kvtierd` and run it. They need a vLLM source tree that has `vllm/v1/kv_offload/tiering`, and a Python environment with vLLM's dependencies. No GPU is needed.

## Layout

```
cmd/kvtierd                     daemon entry point
pkg/kvtier/pool                      shared-memory arena, block index, pins, eviction
pkg/kvtier/policy                    eviction policies
pkg/kvtier/peer                      block transfer between daemons
pkg/kvtier/protocol                  daemon wire format
pkg/kvtier/server                    Unix socket server
kv_connectors/llmd_kv_tiering        engine shims: vLLM on GPU, router hints
kv_connectors/llmd_kv_tiering/docs   design, roadmap and results
```
