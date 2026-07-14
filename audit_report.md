# Audit Report — vLLM Optimization Configuration

> Generated: 2026-07-14
> Based on: vLLM v0.22.1 (`releases/v0.22.1`)

---

## docker-compose.yml

| Flag | Value | Status | Source |
|---|---|---|---|
| `--model` | `/model` | ✅ Valid | `arg_utils.py:780` |
| `--served-model-name` | `Qwen3.5-2B` | ✅ Valid | `arg_utils.py:832` |
| `--host` | `0.0.0.0` | ✅ Valid | `cli_args.py:228` |
| `--port` | `8000` | ✅ Valid | `cli_args.py:230` |
| `--max-model-len` | `262144` | ✅ Valid | `arg_utils.py:802` |
| `--gpu-memory-utilization` | `0.95` | ✅ Valid | `arg_utils.py:1119` |
| `--kv-cache-dtype` | `fp8_e4m3` | ✅ Valid | `arg_utils.py:1124`, `cache.py:23` |
| `--quantization` | `fp8` | ✅ Valid | `arg_utils.py:803`, `quantization/__init__.py:14` |
| `--enable-prefix-caching` | *(flag)* | ✅ Valid | `arg_utils.py:1129` |
| `--max-num-seqs` | `1024` | ✅ Valid | `arg_utils.py:1367`, `scheduler.py:42-44` |
| `--max-num-batched-tokens` | `8192` | ✅ Valid | `arg_utils.py:1360`, `scheduler.py:42-44` |
| `--optimization-level` | `O3` | ✅ Valid | `arg_utils.py:1491`, `vllm.py:72-84` |
| `--block-size` | `32` | ✅ Valid | `arg_utils.py:1117`, `cache.py:45` |

**Verdict:** PASS — all flags valid.

---

## docker-compose.local.yml

All flags same as above, except `max-model-len=8192` and `gpu-memory-utilization=0.90` for local testing.

**Verdict:** PASS — all flags valid.

---

## Dockerfile.custom

| Env Var / Instruction | Value | Status | Issue |
|---|---|---|---|
| `VLLM_ATTENTION_BACKEND` | `FLASHINFER` | ❌ **NO-OP** | Not defined in `vllm/envs.py`. Correct way: `--attention-backend=flashinfer` CLI flag (`arg_utils.py:900`). No env var with this name exists anywhere in vllm. |
| `VLLM_USE_V1` | `0` | ❌ **NO-OP** | Not defined in `vllm/envs.py`. No env var with this name exists anywhere in vllm v0.22.1. V1 engine selection is not controlled by this env var. |
| `TORCHINDUCTOR_FX_REMOTE_CACHE` | `1` | ❌ **NO-OP** | Not defined in `vllm/envs.py`. This is a PyTorch Inductor env var that vLLM does not read or use. |
| `flashinfer-python` pip install | *(package)* | ⚠️ **LOW VALUE** | FlashInfer attention backend can be activated via `--attention-backend=flashinfer` CLI flag (`arg_utils.py:900`). However this requires flashinfer to be installed. In v0.22.1 the V0 code path does not use V1 attention backends. FlashInfer is only a V1 backend. |
| `ninja` pip install | *(package)* | ⚠️ **LOW VALUE** | Ninja build tool. Only relevant if compiling CUDA kernels from source. Pre-built wheel (`VLLM_USE_PRECOMPILED=1`) does not need ninja at runtime. |
| `xgrammar` pip install | *(package)* | ❌ **UNNECESSARY** | xgrammar is only needed for structured output / grammar-constrained generation. Not relevant for ERS/GPQA benchmarking. |
| `--no-deps` flag | *(pip flag)* | ⚠️ **RISK** | `--no-deps` skips dependency resolution. FlashInfer and xgrammar may have missing runtime dependencies. |

**Verdict:** FAIL — 3 no-op env vars, 2 low-value packages, 1 risky install flag.

### Fixes Applied
1. Removed `VLLM_ATTENTION_BACKEND`, `VLLM_USE_V1`, `TORCHINDUCTOR_FX_REMOTE_CACHE`
2. Removed `ninja` and `xgrammar` packages
3. Removed `--no-deps` flag
4. Added `flashinfer-python` as optional install
5. Added `--attention-backend=flashinfer` as a CLI recommendation instead of env var

---

## optimization_configs.py

| Field | Value | Status | Issue |
|---|---|---|---|
| `use_v2_block_manager: bool` | `True` | ❌ **BROKEN** | Flag `use_v2_block_manager` does NOT exist in `arg_utils.py`. Not in `vllm/config/scheduler.py` either. V2 block manager is the default and only option in v0.22.1. |
| `performance_mode: str = "balanced"` | `"balanced"` | ✅ Valid | Defined in `vllm.py:359`, type `PerformanceMode = Literal["balanced", "interactivity", "throughput"]` |
| `performance_mode=None` (quant_only preset) | `None` | ❌ **TYPE ERROR** | `PerformanceMode` is `Literal["balanced", "interactivity", "throughput"]`. `None` is not assignable. Will cause pydantic validation error. |
| Comments about `scheduler_delay_factor` | *(comment)* | ❌ **MISLEADING** | `scheduler_delay_factor` does NOT exist as a CLI flag in `arg_utils.py` or in `vllm/config/scheduler.py`. |
| Comments about `num_scheduler_steps` | *(comment)* | ❌ **MISLEADING** | `num_scheduler_steps` does NOT exist as a CLI flag in `arg_utils.py`. |

**Verdict:** FAIL — 1 broken field, 1 type error, 2 misleading comments.

---

## benchmark_ers.py

| Line | Code | Status | Issue |
|---|---|---|---|
| 15 | `import urllib.error` | ⚠️ **UNUSED** | `urllib.error` not used (exception handling uses `Exception` catch-all). |
| 18 | `from typing import Optional` | ⚠️ **UNUSED** | Not needed — Python 3.12+ native `X | None` is used. |
| Optional | `request_id: str` missing Optional | ⚠️ **POTENTIAL ISSUE** | `request_id` annotated `str` but initialised via `req.get("id", ...)` which could return `None` |

**Verdict:** PASS with minor cleanup.

---

## benchmark_gpqa.py

| Line | Code | Status | Issue |
|---|---|---|---|
| 16 | `import statistics` | ❌ **UNUSED** | `statistics` import not used anywhere in the file. |

**Verdict:** PASS with cleanup.

---

## Summary

| File | Status | Critical | Warning |
|---|---|---|---|
| docker-compose.yml | ✅ PASS | 0 | 0 |
| docker-compose.local.yml | ✅ PASS | 0 | 0 |
| Dockerfile.custom | ❌ FAIL | 3 no-op env vars | 2 unnecessary packages |
| optimization_configs.py | ❌ FAIL | 1 broken field, 1 type error | 2 misleading comments |
| benchmark_ers.py | ✅ PASS | 0 | 2 unused imports |
| benchmark_gpqa.py | ✅ PASS | 0 | 1 unused import |
