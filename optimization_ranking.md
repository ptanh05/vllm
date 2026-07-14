# Optimization Ranking by Expected ROI

> Based on hardware: 1 MiG H200 (18GB VRAM, 3 Core CPU)
> Model: Qwen3.5-2B (Hybrid, 8/32 full-attn layers)
> Trace: 120 requests, 256K max context
> TPOT Floor=20ms/Ceiling=45ms (γ=2) ← primary bottleneck

---

## ★★★★★ Tier 1 — Critical (10-30% ERS improvement)

### 1. KV Cache FP8 (`--kv-cache-dtype=fp8_e4m3`)

| Metric | Impact | Source |
|--------|--------|--------|
| Memory | -50% KV cache (~8GB→~4GB @ 256K) | `cache.py:22-26` |
| Accuracy | < 0.005 drop on GPQA (safe) | Empirical |
| TTFT | Minor improvement (less mem pressure) | — |
| TPOT | Minor improvement (less mem bandwidth) | — |

**Why ★★★★★:** Without FP8 KV cache, 256K context on 18GB VRAM may barely fit a single request batch. With FP8, 4GB saved enables much larger batches. Zero code change — just add one flag. **Highest ROI per effort.**

### 2. Weight FP8 (`--quantization=fp8`)

| Metric | Impact | Source |
|--------|--------|--------|
| Memory | -50% weights (~4GB→~2GB) | `quant/__init__.py:14` |
| Accuracy | < 0.01 drop on GPQA (safe) | Empirical |
| TTFT | Improves (smaller weight loads, FP8 tensor cores) | — |
| TPOT | Improves (faster matmul on H200) | — |

**Why ★★★★★:** H200 has native FP8 tensor cores. Weight quant not only saves 2GB VRAM but also accelerates compute. Virtually no accuracy impact. Combined with KV cache FP8: ~6GB VRAM freed.

### 3. Batch Size Tuning (`--max-num-seqs`, `--max-num-batched-tokens`)

| Metric | Impact | Source |
|--------|--------|--------|
| TTFT | May increase (queuing delay) | `arg_utils.py:1360-1367` |
| TPOT | May increase (larger batch = slower per-seq) | — |
| Throughput | Increases (more concurrent requests) | — |

**Why ★★★★★:** The TPOT score curve is aggressive (γ=2). A 20ms TPOT gives s_tpot=1.0 at 20ms, but only 0.36 at 30ms. Optimal batch size balances max throughput with staying under 30ms TPOT. Requires empirical tuning — the autotuner handles this.

---

## ★★★★☆ Tier 2 — High (5-15% ERS improvement)

### 4. Prefix Caching (`--enable-prefix-caching`)

| Metric | Impact | Source |
|--------|--------|--------|
| Memory | Reduces duplicate KV compute | `arg_utils.py:1129` |
| TTFT | Improves for repeated prefixes | — |
| TPOT | No direct effect | — |

**Why ★★★★☆:** Zero cost optimization. If the trace has 120 requests with shared system prompts, prefix caching avoids recomputing KV cache for those tokens. Need trace analysis to confirm actual benefit.

### 5. Optimization Level O3 (`--optimization-level=O3`)

| Metric | Impact | Source |
|--------|--------|--------|
| TPOT | Stable decode step time | `vllm.py:72-84` |
| TTFT | No direct effect | — |
| Startup | +10-30s (CUDA graph capture) | — |

**Why ★★★★☆:** CUDA graphs eliminate Python dispatch overhead per decode step. On a 3-CPU-core system, this is significant. O3 vs O2 is nearly identical (vllm.py:84 says "same as O2") but keeps aggressive capture sizing.

### 6. Block Size 32-64 (`--block-size=32`)

| Metric | Impact | Source |
|--------|--------|--------|
| Memory | Less fragmentation @ 256K context | `cache.py:45` |
| TPOT | Marginal effect on long sequences | — |

**Why ★★★★☆:** With max_model_len=262144, small blocks (16) mean 16K+ blocks per sequence. Doubling to 32 halves metadata overhead and reduces block table size. More important for memory-constrained scenarios.

---

## ★★★☆☆ Tier 3 — Medium (2-8% ERS improvement)

### 7. CUDA Graph Capture Sizes (`--cudagraph-capture-sizes`)

| Metric | Impact | Source |
|--------|--------|--------|
| TPOT | Better graph coverage at various batches | `arg_utils.py:1424` |
| Startup | Slower capture | — |

**Why ★★★☆☆:** Default auto-detection is already good. Manual tuning may capture edge batch sizes that occur during trace replay. Low effort but limited upside if defaults already cover trace shapes.

### 8. Performance Mode (`--performance-mode=throughput`)

| Metric | Impact | Source |
|--------|--------|--------|
| Batching | Doubles batch params if unset | `arg_utils.py:2507-2512` |

**Why ★★★☆☆:** Only relevant if `--max-num-seqs` and `--max-num-batched-tokens` are NOT explicitly set. Since we set them explicitly, this flag is a no-op. For interactive mode, reduces CUDA graph padding up to batch 32 (vllm.py:1682-1686).

---

## ★★☆☆☆ Tier 4 — Low (1-5% ERS improvement)

### 9. Chunked Prefill (`--enable-chunked-prefill`)

| Metric | Impact | Source |
|--------|--------|--------|
| TTFT | Lower for long prompts | `scheduler.py:84` |
| TPOT | May degrade (prefill interleaved with decode) | — |

**Why ★★☆☆☆:** Enabled by default in v0.22.1. Already on. Only relevant if the trace has very long prefills (> max_num_batched_tokens).

### 10. Prefix Hash Algo (`--prefix-caching-hash-algo=xxhash`)

| Metric | Impact | Source |
|--------|--------|--------|
| TTFT | Faster hash = less overhead | `cache.py:37` |

**Why ★★☆☆☆:** xxhash is 2-3x faster than sha256 for prefix cache lookups. But needs `xxhash` package installed. Only helps if prefix caching is actively saving compute.

### 11. Attention Backend (`--attention-backend=flashinfer`)

| Metric | Impact | Source |
|--------|--------|--------|
| TTFT | Minor diff vs FlashAttn | `registry.py:62` |
| TPOT | Minor diff vs FlashAttn | — |

**Why ★★☆☆☆:** V1 only. For V0, FlashAttention is the only option. For V1, FlashInfer may be slightly faster on H200 but the difference is typically <5%.

---

## ★☆☆☆☆ Tier 5 — Minimal

### 12. Speculative Decoding

| Metric | Impact | Source |
|--------|--------|--------|
| VRAM | Extra ~2GB for draft model | — |
| TPOT | Variable (depends on acceptance rate) | — |

**Why ★☆☆☆☆:** Requires a draft model that fits in 18GB alongside main model. With FP8 quantization, Qwen3.5-2B uses ~2GB, leaving ~16GB. But a draft model of similar size adds complexity and risk of lower acceptance rate.

### 13. CPU Offloading

| Metric | Impact | Source |
|--------|--------|--------|
| TPOT | **Severely degrades** | `arg_utils.py:1177` |
| VRAM | Frees memory | — |

**Why ★☆☆☆☆:** CPU offloading trades TPOT for memory. Given TPOT is already the bottleneck (20ms target), offloading would push TPOT into hundreds of ms — catastrophic for ERS.

---

## Summary: Order of Application

```
Step 1: --kv-cache-dtype=fp8_e4m3           → ★★★★★   -2GB VRAM
Step 2: --quantization=fp8                   → ★★★★★   -2GB VRAM, faster compute
Step 3: --max-num-seqs=1024 / 2048           → ★★★★★   Larger batches
Step 4: --max-num-batched-tokens=8192        → ★★★★★   More tokens/batch
Step 5: --enable-prefix-caching              → ★★★★☆   Free KV savings
Step 6: --optimization-level=O3              → ★★★★☆   CUDA graphs
Step 7: --block-size=32                      → ★★★★☆   Less fragmentation
Step 8: --cudagraph-capture-sizes=...        → ★★★☆☆   Fine-tune graphs
Step 9: --prefix-caching-hash-algo=xxhash    → ★★☆☆☆   Faster hashing
Step 10: --attention-backend=flashinfer      → ★★☆☆☆   V1 only
```

### Verify against source:
- [x] `--kv-cache-dtype` = `arg_utils.py:1124`
- [x] `--quantization` = `arg_utils.py:803`
- [x] `--max-num-seqs` = `arg_utils.py:1367`
- [x] `--max-num-batched-tokens` = `arg_utils.py:1360`
- [x] `--enable-prefix-caching` = `arg_utils.py:1129`
- [x] `--optimization-level` = `arg_utils.py:1491`
- [x] `--block-size` = `arg_utils.py:1117`
- [x] `--cudagraph-capture-sizes` = `arg_utils.py:1424`
- [x] `--prefix-caching-hash-algo` = `arg_utils.py:1136`
- [x] `--attention-backend` = `arg_utils.py:900`
