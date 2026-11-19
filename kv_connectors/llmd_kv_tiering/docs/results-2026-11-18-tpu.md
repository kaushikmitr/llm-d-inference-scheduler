# TPU check, 2026-11-18: 2 single-host TPU7x replicas, Qwen3-8B

This is the first run of `llmd_kv_tiering.tpu_connector` on TPU hardware.

## Setup

- GKE tpu7x pool with a 2x2x4 slice. Each replica runs on one host: 4 chips, 8 JAX devices, TP=8.
- `vllm/vllm-tpu:v0.27.0` (vLLM 0.26.1rc1 with tpu-inference), Qwen3-8B, fp8 KV cache, 256-token blocks, 2,000 host-memory chunks.
- `kv_connector` is `TPUOffloadConnector`, loaded from `llmd_kv_tiering.tpu_connector` through `kv_connector_module_path`. tpu-inference is not patched.
- Each pod runs a `kvtierd` sidecar with a 32 GiB pool. One chunk is `(1, 36, 256, 8, 4, 128)` fp8, 37.7 MB.
- Requests go through vLLM's OpenAI HTTP API. Hints travel in `kv_transfer_params["kv_hints"]`.

## Pulling a prefix from the other replica

Replica A computes a prompt. Replica B then serves the same prompt, either with no hint (recompute) or with a `kv.fetch` hint naming A's daemon (placeholder hash, opportunistic). The prompt is 11,584 random tokens with one output token. Each figure is the median of 5 runs.

| | Time on B | Tokens cached on B |
| --- | --- | --- |
| Recompute (no hint) | 332 ms | 0 |
| Pull from A (hint) | 255 ms | 11,520 (every full block) |

## Correctness

Each prompt was a 9,000-word text plus a question, with 48 greedy output tokens. A answered it fresh, then again from its own HBM prefix cache. B answered it with KV pulled from A.

| Comparison | Identical outputs |
| --- | --- |
| A's own prefix-cache hit vs B with pulled KV | 8 of 8 |
| A fresh vs A's own prefix-cache hit | 2 of 8 |
| A fresh vs B with pulled KV | 2 of 8 |

So a pulled prefix gives exactly the output that a native prefix-cache hit gives. Both differ from a fresh prefill late in some generations. That is because prefix caching on this engine is not bit-exact. A cache hit computes only the tail, through different kernel shapes. The tier does not cause it.

## Engine restart

Only A's vLLM container was restarted. The `kvtierd` sidecar and its pool kept running. The restarted scheduler attached and found 910 blocks in the pool. With no hints:

| Prompt, 9,011 tokens | Time | Tokens cached |
| --- | --- | --- |
| Computed before the restart (3 prompts) | 0.10 to 0.22 s | 8,960 |
| Never seen (3 prompts) | 0.30 to 0.33 s | 0 |

## Limits

- Single-host replicas only. With several hosts, each host holds a shard of every chunk, and a block may count as a hit only when every host's daemon holds it. The connector refuses to start on more than one host.
- A fetch hint makes the scheduler pull the blocks before it answers the lookup, so the pull blocks the scheduler for that time. The TPU connector's own loads already block each step.
- The connector depends on private names in tpu-inference (`offload_manager`, `cpu_backend`, `chunk_pool`). A tpu-inference upgrade can break it.
