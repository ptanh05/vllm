"""
Optimization configuration system for vLLM serving.

Defines the full search space of optimizations available through
CLI flags and internal vLLM config patches.

Usage:
    from optimization.config import OptConfig, get_optimized_args

    cfg = OptConfig.baseline()
    cfg.kv_cache_dtype = "fp8_e4m3"
    cfg.performance_mode = "throughput"
    args = cfg.to_command_args()

All flags verified against vllm/engine/arg_utils.py and config/*.py.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


@dataclass
class OptConfig:
    """Complete optimization configuration."""

    # ── Core model ──
    model: str = "/model"
    served_model_name: str = "Qwen3.5-2B"
    host: str = "0.0.0.0"
    port: int = 8000
    max_model_len: int = 262144
    tensor_parallel_size: int = 1

    # ── Memory ──
    # Verify: arg_utils.py:1119, cache.py:field auto-generated
    gpu_memory_utilization: float = 0.95
    # Verify: arg_utils.py:1124, cache.py:22-26 (valid: auto, fp8, fp8_e4m3, fp8_e5m2)
    kv_cache_dtype: str = "auto"
    # Verify: arg_utils.py:1142, cache.py:112
    kv_cache_dtype_skip_layers: str | None = None

    # ── Batching ──
    # Verify: arg_utils.py:1367, scheduler.py:63
    max_num_seqs: int | None = None  # None = auto
    # Verify: arg_utils.py:1360, scheduler.py:49
    max_num_batched_tokens: int | None = None  # None = auto

    # ── Block management ──
    # Verify: arg_utils.py:1117, cache.py:45 (default=16)
    block_size: int = 16

    # ── Prefix caching ──
    # Verify: arg_utils.py:1129
    enable_prefix_caching: bool = True
    # Verify: arg_utils.py:1136, cache.py:37 (valid: sha256, sha256_cbor, xxhash, xxhash_cbor)
    prefix_caching_hash_algo: str = "sha256"  # xxhash is faster but needs extra pkg

    # ── Performance mode ──
    # Verify: vllm.py:87, arg_utils.py:1493
    performance_mode: Literal["balanced", "interactivity", "throughput"] = "balanced"

    # ── Compilation ──
    # Verify: vllm.py:72-84, arg_utils.py:1491
    optimization_level: Literal["O0", "O1", "O2", "O3"] = "O2"

    # ── Scheduling ──
    # Verify: arg_utils.py:1392, scheduler.py:84
    enable_chunked_prefill: bool = True
    # NOTE: scheduler_delay_factor does NOT exist as a vLLM CLI flag
    # Verified: grepped arg_utils.py, scheduler.py — no match
    # scheduler_policy also does NOT exist as a CLI flag

    # ── Quantization ──
    # Verify: arg_utils.py:803, quantization/__init__.py:13 (valid: fp8, awq, gptq...)
    quantization: str | None = None

    # ── Speculative decoding ──
    spec_model: str | None = None
    spec_tokens: int | None = None

    # ── CUDA Graphs ──
    # Verify: arg_utils.py:1424, type list[int] | None
    # CLI format: --cudagraph-capture-sizes=1,2,4,8,16 (NOT -cc. prefix)
    cudagraph_capture_sizes: str | None = None

    # ── Attention backend ──
    # Verify: arg_utils.py:900, v1/attention/backends/registry.py:34-80
    # Valid: FLASH_ATTN, FLASHINFER, TRITON_ATTN (V1 only)
    attention_backend: str | None = None

    @staticmethod
    def baseline() -> "OptConfig":
        """Return the competition baseline configuration."""
        return OptConfig(
            gpu_memory_utilization=0.95,
            max_model_len=262144,
            enable_prefix_caching=True,
        )

    @staticmethod
    def recommended() -> "OptConfig":
        """Return the recommended optimized configuration."""
        return OptConfig(
            gpu_memory_utilization=0.95,
            max_model_len=262144,
            enable_prefix_caching=True,
            prefix_caching_hash_algo="xxhash",
            kv_cache_dtype="fp8_e4m3",
            quantization="fp8",
            max_num_seqs=1024,
            max_num_batched_tokens=None,
            block_size=32,
            performance_mode="throughput",
            optimization_level="O3",
            enable_chunked_prefill=True,
            cudagraph_capture_sizes="1,2,4,8,16,32,64,128,256",
        )

    @staticmethod
    def aggressive() -> "OptConfig":
        """More aggressive: push GPU harder, accept higher risk of OOM."""
        return OptConfig(
            gpu_memory_utilization=0.97,
            max_model_len=262144,
            enable_prefix_caching=True,
            prefix_caching_hash_algo="xxhash",
            kv_cache_dtype="fp8_e4m3",
            kv_cache_dtype_skip_layers=None,
            quantization="fp8",
            max_num_seqs=2048,
            max_num_batched_tokens=None,
            block_size=64,
            performance_mode="throughput",
            optimization_level="O3",
            enable_chunked_prefill=True,
            cudagraph_capture_sizes="1,2,4,8,16,32,64,128,256,512",
        )

    def to_command_args(self) -> list[str]:
        """Convert config to vLLM CLI argument list."""
        args = [
            "--model", self.model,
            "--served-model-name", self.served_model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            f"--gpu-memory-utilization={self.gpu_memory_utilization}",
        ]

        if self.kv_cache_dtype != "auto":
            args.append(f"--kv-cache-dtype={self.kv_cache_dtype}")

        if self.kv_cache_dtype_skip_layers:
            args.append(f"--kv-cache-dtype-skip-layers={self.kv_cache_dtype_skip_layers}")

        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")
            if self.prefix_caching_hash_algo != "sha256":
                args.append(f"--prefix-caching-hash-algo={self.prefix_caching_hash_algo}")

        if self.quantization:
            args.append(f"--quantization={self.quantization}")

        if self.max_num_seqs is not None:
            args.append(f"--max-num-seqs={self.max_num_seqs}")

        if self.max_num_batched_tokens is not None:
            args.append(f"--max-num-batched-tokens={self.max_num_batched_tokens}")

        args.append(f"--block-size={self.block_size}")
        args.append(f"--performance-mode={self.performance_mode}")
        args.append(f"--optimization-level={self.optimization_level}")

        if self.enable_chunked_prefill:
            args.append("--enable-chunked-prefill")

        if self.cudagraph_capture_sizes:
            args.append(f"--cudagraph-capture-sizes={self.cudagraph_capture_sizes}")

        if self.attention_backend:
            args.append(f"--attention-backend={self.attention_backend}")

        if self.spec_model:
            args.append(f"--spec-model={self.spec_model}")
        if self.spec_tokens:
            args.append(f"--num-speculative-tokens={self.spec_tokens}")

        return args

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESETS: dict[str, OptConfig] = {
    "baseline": OptConfig.baseline(),
    "recommended": OptConfig.recommended(),
    "aggressive": OptConfig.aggressive(),
}

# ── Tuning priority, from highest impact to lowest ──

TUNING_PRIORITY = [
    # Tier 1 — memory (highest impact on 18GB VRAM)
    "kv_cache_dtype",
    "quantization",
    "gpu_memory_utilization",

    # Tier 2 — batching & throughput
    "max_num_seqs",
    "max_num_batched_tokens",
    "performance_mode",

    # Tier 3 — latency & scheduling
    "optimization_level",
    "block_size",
    "enable_chunked_prefill",

    # Tier 4 — fine-tuning
    "cudagraph_capture_sizes",
    "prefix_caching_hash_algo",
    "attention_backend",
    "enable_prefix_caching",
]
