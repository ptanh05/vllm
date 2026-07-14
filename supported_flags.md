# Supported vLLM CLI Flags — AI RACE Phase 1

> Source verification: `vllm/engine/arg_utils.py`, `vllm/config/`

---

## Legend

| Icon | Meaning |
|------|---------|
| ⭐ | High ROI for Phase 1 |
| 🅃 | Affects TTFT |
| 🅃₂ | Affects TPOT |
| 💾 | Affects Memory/VRAM |
| 🎯 | Affects Accuracy |

---

## 1. Core Model

| Flag | Default | Source | Description | Impact |
|---|---|---|---|---|
| `--model` | *(required)* | `arg_utils.py:780` | Path to model weights on disk or HF repo ID | — |
| `--served-model-name` | model name | `arg_utils.py:832` | Model name exposed in API | — |
| `--max-model-len` | model default | `arg_utils.py:802` | Maximum sequence length (prompt + generation) | 🅃 🅃₂ 💾 |
| `--tensor-parallel-size` | 1 | `arg_utils.py:959` | Number of GPUs for tensor parallelism (fixed=1 for MiG) | 💾 |

---

## 2. Memory & Cache

| Flag | Default | Source | Description | Impact |
|---|---|---|---|---|
| **⭐** `--gpu-memory-utilization` | `0.90` | `arg_utils.py:1119` `cache.py` | Fraction of VRAM to allocate for model + KV cache. Higher = bigger KV cache = larger batches. Too high → OOM. | 🅃 🅃₂ 💾 |
| **⭐** `--kv-cache-dtype` | `"auto"` | `arg_utils.py:1124` `cache.py:22-26` | KV cache precision. Valid: `"auto"`, `"fp8"`, `"fp8_e4m3"`, `"fp8_e5m2"`. FP8 cuts KV cache memory by ~50% with negligible accuracy loss. | 🅃 🅃₂ 💾 🎯 |
| `--kv-cache-dtype-skip-layers` | None | `arg_utils.py:1142` `cache.py:112` | Skip FP8 quantization for specific layer names. Preserves accuracy at cost of memory. | 💾 🎯 |
| `--block-size` | `16` | `arg_utils.py:1117` `cache.py:45` | KV cache block size in tokens. Larger = less fragmentation but coarser allocation. 32 or 64 beneficial for long contexts. | 💾 🅃₂ |
| `--num-gpu-blocks-override` | auto | `arg_utils.py:1126` | Force specific number of GPU KV cache blocks. Advanced tuning only. | 💾 |

---

## 3. Quantization

| Flag | Default | Source | Description | Impact |
|---|---|---|---|---|
| **⭐** `--quantization` | None | `arg_utils.py:803` `quant/__init__.py:13` | Weight quantization method. `"fp8"` reduces weights ~50%, leverages H200 FP8 tensor cores for faster matmul. Valid: awq, gptq, fp8, etc. | 💾 🅃 🅃₂ 🎯 |
| `--quantization-config` | None | `arg_utils.py:805` | JSON/dict config for quant method. Advanced, rarely needed. | 🎯 |
| `--enforce-eager` | `False` | `arg_utils.py:811` | Disable CUDA graphs. HUGE TPOT penalty. Only for debugging. | 🅃₂ |

---

## 4. Batching & Scheduling

| Flag | Default | Source | Description | Impact |
|---|---|---|---|---|
| **⭐** `--max-num-seqs` | `128` (scheduler default) | `arg_utils.py:1367` `scheduler.py:44` | Max sequences per iteration. Larger = higher throughput but worse TPOT. Tune to VRAM. | 🅃 🅃₂ 💾 |
| **⭐** `--max-num-batched-tokens` | `2048` | `arg_utils.py:1360` `scheduler.py:42` | Max tokens per iteration (prefill + decode). Larger = more work per step. | 🅃 🅃₂ |
| `--enable-chunked-prefill` | `True` | `arg_utils.py:1392` `scheduler.py:84` | Split long prefills into chunks to mix with decode. Reduces TTFT for long prompts at cost of throughput. | 🅃 🅃₂ |
| `--disable-chunked-mm-input` | `False` | `scheduler.py:117` | Only for multimodal models. Not relevant for Qwen3.5-2B. | — |
| `--num-scheduler-steps` | `1` | `vllm.py` | **(Not in arg_utils CLI)** Number of schedule steps per engine step. V1 only. | 🅃₂ |

---

## 5. Prefix Caching

| Flag | Default | Source | Description | Impact |
|---|---|---|---|---|
| **⭐** `--enable-prefix-caching` | `False` | `arg_utils.py:1129` | Reuse KV cache blocks across requests with shared prompt prefixes. Free optimization if trace has repeated prefixes. | 🅃 🅃₂ 💾 |
| `--prefix-caching-hash-algo` | `"sha256"` | `arg_utils.py:1136` `cache.py:37` | Hash algorithm for prefix cache keys. `"xxhash"` is 2-3x faster but needs `xxhash` package. Valid: sha256, sha256_cbor, xxhash, xxhash_cbor. | 🅃 🅃₂ |
| `--hash-block-size` | auto | `cache.py:54` | Block size for hash computation. Typically = block_size. Rarely tuned. | 💾 |

---

## 6. CUDA Graphs

| Flag | Default | Source | Description | Impact |
|---|---|---|---|---|
| **⭐** `--optimization-level` | `O2` | `arg_utils.py:1491` `vllm.py:72-84` | Compilation and graph capture aggressiveness. O0=no graphs, O1=basic, O2=full, O3=same as O2. Higher = better TPOT after warmup. | 🅃₂ |
| `--cudagraph-capture-sizes` | auto | `arg_utils.py:1424` | Comma-separated batch sizes for CUDA graph capture. E.g. "1,2,4,8,16,32,64". More sizes = better coverage but slower startup. | 🅃₂ |
| `--max-cudagraph-capture-size` | `512` | `arg_utils.py:1427` | Maximum batch size for CUDA graph capture. | 🅃₂ |
| `--cudagraph-metrics` | `False` | `arg_utils.py:1337` | Enable CUDA graph metrics endpoint. For profiling, not performance. | — |

---

## 7. Performance Mode

| Flag | Default | Source | Description | Impact |
|---|---|---|---|---|
| `--performance-mode` | `"balanced"` | `arg_utils.py:1493` `vllm.py:87` | `"balanced"`: default. `"throughput"`: doubles batch params **only if unset** (arg_utils.py:2507-2512). `"interactivity"`: fine-grained CUDA graphs up to batch 32 (vllm.py:1682-1686). | 🅃 🅃₂ |

---

## 8. Attention

| Flag | Default | Source | Description | Impact |
|---|---|---|---|---|
| `--attention-backend` | auto | `arg_utils.py:900` `v1/attention/backends/registry.py:34-80` | Attention backend. Options: FLASH_ATTN, FLASHINFER, TRITON_ATTN, etc. **V1 only** — V0 uses FlashAttn directly. | 🅃 🅃₂ |

---

## 9. Advanced

| Flag | Default | Source | Description | Impact |
|---|---|---|---|---|
| `--spec-model` | None | `arg_utils.py:1464` | Draft model for speculative decoding. Requires extra VRAM. | 🅃 🅃₂ 💾 |
| `--num-speculative-tokens` | (spec config) | `arg_utils.py:1470` | Number of speculative tokens. | 🅃 🅃₂ 💾 |
| `--spec-method` | None | `arg_utils.py:1463` | Speculative decoding method. | 🅃 🅃₂ 💾 |
| `--cpu-offload-gb` | `0` | `arg_utils.py:1177` | Offload KV cache to CPU. Saves VRAM but hurts TPOT. | 💾 🅃₂ |
| `--load-format` | auto | `arg_utils.py:871` | Model weight format. `"safetensors"` for fast loading. | — |
| `--dtype` | auto | `arg_utils.py:788` | Model dtype. `"auto"` reads from config. Use `"bfloat16"` for full accuracy baseline. | 💾 🎯 |

---

## 10. Flags That Do NOT Exist (Removed)

| Flag | Reason |
|---|---|
| `--scheduler-delay-factor` | Not in `arg_utils.py` or any config. Never existed in v0.22.1. |
| `--use-v2-block-manager` | Not in `arg_utils.py`. V2 is the default and only option in v0.22.1. |
| `--scheduler-policy` | Not exposed as CLI flag. Internal only (`scheduler.py:109`). |
| `VLLM_ATTENTION_BACKEND` (env) | Not in `vllm/envs.py`. No env var with this name exists. Use `--attention-backend` CLI flag instead. |
| `VLLM_USE_V1` (env) | Not in `vllm/envs.py`. No env var with this name exists. |
