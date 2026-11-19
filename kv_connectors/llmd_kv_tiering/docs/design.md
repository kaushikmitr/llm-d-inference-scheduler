# Design

## Goal

Give llm-d a node-level KV tier that outlives engines, is shared by every engine on the node, moves blocks between nodes when the router asks, and runs on GPU and TPU. llm-d controls the hint actions the router can send, and can add new ones as its routing needs them.

## Architecture

```
            llm-d router (EPP)
                 | hints, in kv_transfer_params
   +-------------+--------------------------+
   | node                                   |
   |  vLLM engine            vLLM engine    |
   |  [GPU HBM]              [GPU HBM]      |
   |      |                      |          |
   |  [primary CPU tier]     [primary CPU]  |   vLLM's OffloadingConnector
   |      |  shim                |  shim    |   llmd_kv_tiering.vllm_tier
   |      +---------+------------+          |
   |                |  Unix socket + shared memory
   |            kvtierd (Go)                |
   |       [shared-memory pool]             |
   +----------------------------------------+
```

vLLM moves KV between GPU and its primary CPU tier. Secondary tiers only ever see host memory. That lets the daemon be a separate process with no GPU code.

On TPU the daemon is the same. Only the engine side differs: the TPU shim plugs into tpu-inference's own offload connector instead of vLLM's tiering framework. See [TPU](#tpu).

**Control plane.** The shim talks to the daemon over a Unix socket. Each frame is a 4-byte length and one msgpack map. Requests: `hello`, `alloc`, `commit`, `pin`, `unpin`, `snapshot`, `stats`. The daemon pushes `event` frames when blocks are stored or evicted.

**Data plane.** The daemon creates the pool with `memfd_create` and passes the file descriptor on the `hello` response with `SCM_RIGHTS`. The shim maps it. A store is one `memmove` from the primary tier into a pool slot. A load is one `memmove` back. The copies run on shim worker threads, and `ctypes.memmove` releases the GIL.

**Lookups are local.** vLLM calls `lookup` on the scheduler thread for every block, so it must not block. The shim keeps a mirror of the keys in its namespace. It fills the mirror from a snapshot at connect and keeps it current from events. The daemon sends each pool change's event, and the reply to the request that caused it, under one lock. So every engine, the one that made the change included, sees changes in the order they happened, and a late event never overrides a newer one. Only events change the mirror. A mirror entry can still be stale for a moment, which costs one failed load that vLLM recomputes.

**Store protocol.** `alloc` reserves slots and evicts if needed. The shim copies, then calls `commit`. A block is invisible to loads until it is committed.

**Load protocol.** `pin` returns slots and protects them from eviction. The shim copies, then calls `unpin`. When a session disconnects, the daemon drops its pins and frees slots it was still writing, after a grace period (`--release-grace`, 10 s by default). The engine may still be copying when its connection drops, for example when the daemon closes an engine that fell behind on events. The grace period keeps a late copy from touching a slot that another engine now owns.

**Timeouts.** A request that times out may still have been applied by the daemon. When a request that changes state (`alloc`, `commit`, `pin`, `unpin`) times out, the shim closes the connection, so the daemon releases whatever the session held. Sends have a deadline too.

**Namespaces.** Engines share a block only when their bytes are interchangeable. Equal byte sizes are not enough, since two engines can lay out the same bytes differently. The default namespace hashes the model, the KV cache dtype, the KV cache groups and their layers, vLLM's KV cache memory layout, the canonical-layout and replicated-layout flags, tokens per hash, blocks per chunk, block size in bytes, TP, PP, PCP, DCP and the vLLM version. The model revision is not in vLLM's offloading config, so pass it as `namespace_salt`.

**Memory accounting.** Pages of a memfd are charged to the memory cgroup of the process that first touches them. The daemon prefaults the pool at start by default, so the pool counts against the daemon's container and not the engine's.

## TPU

vLLM on TPU does not use vLLM's tiering framework. tpu-inference has its own `TPUOffloadConnector`, which keeps chunks in host memory in its own store. The TPU shim, `llmd_kv_tiering.tpu_connector`, subclasses that connector. vLLM loads it through `kv_connector_module_path`, so tpu-inference is not patched.

```
            llm-d router (EPP)
                 | hints, in kv_transfer_params
   +-------------+---------------------------------------------+
   | TPU host                                                  |
   |   scheduler (EngineCore)            worker                |
   |   [chunk index: hash -> chunk id]   [TPU HBM]             |
   |        |  pool hits, pins,              |                 |
   |        |  virtual chunk ids ---------> [host chunk store] |
   |        |                                 |  copy on save  |
   |        +-------------+-------------------+                |
   |                      |  Unix socket + shared memory       |
   |                  kvtierd (Go)                             |
   |             [shared-memory pool]                          |
   +-----------------------------------------------------------+
```

The connector splits its work between a scheduler, which knows block hashes, and a worker, which holds the data. The shim hooks both.

**Save.** Every chunk the worker stores in host memory is also copied into the pool, on a background thread. The scheduler sends the worker the hash of each chunk it saves, in the step's metadata.

**Lookup.** The scheduler's chunk index is wrapped. Blocks found in the pool count as hits once the host-memory hits run out. The scheduler pins each one in the daemon, so it cannot be evicted before the load runs. It then gives the block a virtual chunk id, at or above 2^40, and sends the pool slot to the worker in the step's metadata.

**Load.** The worker's chunk store is wrapped. For a virtual chunk id it returns the pool slot's bytes as a numpy array, with the chunk's shape and dtype. The connector's own host-to-device copy and scatter then run unchanged. When the load is reported done, the scheduler unpins the block.

**Fetch hints.** A `kv.fetch` hint in `kv_transfer_params["kv_hints"]` makes the scheduler pull the request's blocks from the named daemon into the local pool before it counts hits. A failed pull only means the blocks are recomputed. A load never starts for a block the local pool does not hold. The pull blocks the scheduler for its duration. The connector's own loads already block each step.

**Chunk layout.** A chunk is one 256-token block stacked across layers: `(1, layers, block, heads, packing, head_dim)` in the KV cache dtype. The worker derives it from the KV cache at start-up, so a cold engine can read the pool before it has saved anything.

**Connector name.** The TPU platform accepts only a fixed list of connector names. The shim exports its subclass as `TPUOffloadConnector`, and vLLM loads that name from the shim's module.

**Single host only, for now.** A replica that spans several hosts keeps one shard of each chunk on each host, so it needs one daemon per host. A block may then count as a hit only when every host's daemon holds it. Otherwise the hosts load different data, or one host skips a collective step and the replica hangs. The shim refuses to start on more than one host until that rule is built.

## Upstream dependencies

| Engine | What we need | Status |
| --- | --- | --- |
| vLLM | Out-of-tree secondary tiers via `module_path` | In vLLM main. vLLM calls the API experimental. |
| vLLM | Hints reaching the tier | Done, no change needed. `ReqContext` carries both `kv_hints` (in-process callers such as Dynamo) and `kv_transfer_params` (the OpenAI HTTP server). The tier reads an envelope from either, so llm-d's router sends hints in `kv_transfer_params["kv_hints"]`. |
| SGLang | A HiCache storage backend | Not started. HiCache has a storage backend interface that a shim could implement. |
| vLLM TPU | A tier hook in tpu-inference | Done for single-host replicas, no change needed. The shim subclasses `TPUOffloadConnector` and loads through `kv_connector_module_path`. It relies on private names in tpu-inference (`offload_manager`, `cpu_backend`, `chunk_pool`), so an upgrade there can break it. A public hook in tpu-inference would remove that risk. |

## Router hints

vLLM's hint envelope is generic: `protocol_version`, `message_id`, and a list of actions, each with `action_type`, `action_version` and a free-form payload. The tier handles four action types:

| Action | Payload | Received by |
| --- | --- | --- |
| `kv.fetch` | `source_endpoint`, `block_hashes` | the engine that needs the blocks |
| `kv.evict_on_fetch` | as `kv.fetch` | the engine that needs the blocks; the source drops them after |
| `kv.prefetch` | `destination_endpoint`, `block_hashes` | the source, which pushes ahead of the request |
| `kv.move` | `destination_endpoint`, `block_hashes` | the source |

These are the action types other KV tiers for vLLM already use (KVCR v0.2.0, version `1.0`), with the same payloads and validation. A router that emits them works with this tier unchanged. New llm-d actions will use their own `llmd.*` prefix. The envelope already allows this with no spec change.

How each action works here:

- **`kv.fetch` and `kv.evict_on_fetch`.** The tier records the hinted hashes for the request. A lookup of a covered block reports a hit. The load asks the local daemon to pull the blocks from the source daemon, then copies them into vLLM as usual. With `evict_on_fetch`, the source drops each block once it has sent it. A hint that names this node's own daemon is ignored, because the blocks are local or gone.
- **`kv.prefetch` and `kv.move`.** These arrive at the source. The source daemon maps the hashes to its keys and offers them to the destination. The destination pulls them. With `move`, the source drops what it sent.
- **Router hashes.** A hash is the low 64 bits of vLLM's block hash, the value vLLM publishes in KV events when `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES` is set. The daemon keeps an index from hash to keys.

For reference, how this tier differs from KVCR v0.2.0, the tier the H200 results compare against:

| | KVCR v0.2.0 | This tier |
| --- | --- | --- |
| Peer data path | NIXL (RDMA) | TCP, straight into pool slots. NIXL is milestone 5. |
| What a peer serves | Engine-owned and KVCR-owned host memory | The daemon's pool only |
| Hint results | Reported to the router | Logged only |
| Endpoint | A KVCR control endpoint per engine rank | One daemon endpoint per node. A multi-host TPU replica will need one per host. |
| TPU | Not supported | Supported for single-host replicas |
| `no_retain` | Rejected | Rejected |
| Fetch failure | Load fails for the missing blocks; vLLM recomputes | Same |

## Roadmap

1. **Done.** Daemon with a shared-memory pool, LRU eviction, pins and session cleanup. vLLM shim with store, load, KV events, engine-restart survival and reconnect. Tested against vLLM's real `TieringOffloadingManager` on CPU.
2. **Done.** Router hints: the four `kv.*` actions, carried in `kv_transfer_params` or `kv_hints`. Peer transfer between daemons over TCP: fetch, evict-on-fetch, prefetch and move. Transfers spread over 8 TCP connections and 8 copy threads. Tested on H200 ([results](results-2026-11-18-h200.md)).
3. **Done for single-host replicas.** TPU shim for tpu-inference: save to the pool, pool hits, peer fetch hints and engine-restart survival. Tested on TPU7x with Qwen3-8B ([results](results-2026-11-18-tpu.md)).
4. Multi-host TPU replicas. One daemon per host. A block counts as a hit only when every host's daemon holds it, and a load is confirmed by every host. Hints name one endpoint per host, and host k pulls from host k of the source replica.
5. A local SSD tier under the pool, and a policy interface that can be swapped at run time.
6. llm-d router integration. The router consumes block events and emits hints. It publishes each pod's daemon endpoints. Prefetch of the next turn.
7. NIXL/RDMA peer transfer through cgo, an object-store tier, and an SGLang shim.

## Open questions

- **Zero-copy stores.** vLLM's primary tier is also a shared-memory file. If the shim passes its file descriptors to the daemon, the daemon can copy, or send to a peer, without a hop through the shim. The files are unlinked after open, so this needs `SCM_RIGHTS` from the shim for each segment.
- **Mirror size.** The snapshot is one message. At about 40 bytes per key, a pool of a million blocks is a 40 MB snapshot. Paging it may be needed.
- **Endpoint discovery.** The router needs each pod's daemon endpoint to write a hint. The tier knows it (`LLMDKVTierManager.endpoint`). It could publish it as a pod label, a KV event field, or a metric.
- **Hint results.** The tier only logs per-action results. The router may want them back for its cost model.
- **Non-blocking TPU fetch.** The TPU scheduler pulls hinted blocks before it answers the lookup. Answering "ask again later" and pulling in the background would stop the pull from stalling other requests. That depends on the TPU vLLM build supporting a deferred match.
- **TPU fetch on the critical path.** The TPU worker copies pool bytes to the device on its main thread, inside the step. Overlapping that copy with the previous step would hide it.
- **Multiple block sizes.** The pool has one slot size today. A node serving two models with different block sizes needs one arena per size.
