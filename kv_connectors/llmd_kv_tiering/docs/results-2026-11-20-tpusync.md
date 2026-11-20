# TPU Sync connector check, 2026-11-20: 2 single-host TPU7x replicas, Qwen3-8B

This is the first run of `llmd_kv_tiering.tpusync_connector`. It is a proof of concept. The connector carries out router KV hints with [TPU Sync](https://github.com/google/tpu-sync) and no kvtierd.

## Setup

- GKE tpu7x pool with a 2x2x4 slice. Each replica runs on one host: 4 chips, 8 JAX devices, TP=8.
- `vllm/vllm-tpu:v0.27.0` (vLLM 0.26.1rc1 with tpu-inference), Qwen3-8B, fp8 KV cache, 256-token blocks. One block is 37.7 MB.
- `tpu-sync-jax` 0.1.1.dev20261120081609 from PyPI. One `global_registry_server` from the `tpu-sync-torch` wheel.
- `kv_connector` is `RaidenOffloadConnector`, loaded from `llmd_kv_tiering.tpusync_connector` through `kv_connector_module_path`. Each store holds 1,500 blocks.
- Requests go through vLLM's OpenAI HTTP API. Hints travel in `kv_transfer_params["kv_hints"]` and name the peer's pod IP.

The image needed three changes to run the wheel. The wheel requires jax 0.11.2 and libtpu 0.0.48, and the image ships jax 0.11.0. flax was raised to 0.12.12 to match. One line of tpu-inference's batched attention kernel was changed from `h.memory_space` to `jax.typeof(h).memory_space`, which jax 0.11.2 requires.

## Results

Each prompt is about 30,000 tokens with 24 greedy output tokens. 29,952 tokens (117 blocks, 4.4 GB) are cacheable. Times are time to first token. Each row is a few single runs, not a benchmark.

| Request on B | Time to first token | Tokens cached | Blocks loaded |
| --- | --- | --- | --- |
| Recompute, no hint | 0.53 s | 0 | 0 |
| `kv.fetch` hint naming A | 0.46 to 0.50 s | 29,952 | 117 from A in 0.34 to 0.42 s |
| After A's `kv.prefetch` hint naming B, no hint on B | 0.12 s | 29,952 | 117 from B's host memory in 0.02 s |
| A's own HBM prefix-cache hit, for reference | 0.09 to 0.14 s | 29,952 | 0 |

Qwen3-8B prefills 30,000 tokens in about half a second on this hardware. A pull that takes 0.4 s therefore saves little here. The pull moved 4.4 GB in 0.34 to 0.42 s.

## Correctness

Each prompt starts with a random 12-digit code and ends by asking for it. A wrong first block gives a wrong answer.

- All 6 requests served from pulled or pushed KV returned the right code.
- In 3 of 3 cases, B's output with KV pulled from A was identical to A's output from its own HBM prefix-cache hit.
- A's fresh output and A's own prefix-cache output differed after the code in 2 of 3 cases. vLLM alone shows this difference, so it is not caused by the connector.

## Other observations

- Without a hint, B did not read from A. This is the default `tpusync_global_lookup: "hint"`.
- 1,872 blocks were saved into a 1,500-block store. No save was refused, and the newest prompt could still be pulled. The store evicts by itself.
- Not tested: a push to a peer whose store is full (TPU Sync documents that the peer refuses it), failures during a load, load under concurrent traffic, and multi-host replicas.
