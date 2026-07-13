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
    gpu_memory_utilization: float = 0.95
    kv_cache_dtype: str = "auto"  # auto, fp8_e4m3, fp8_e5m2
    kv_cache_dtype_skip_layers: str | None = None

    # ── Batching ──
    max_num_seqs: int | None = None  # None = auto
    max_num_batched_tokens: int | None = None  # None = auto

    # ── Block management ──
    block_size: int = 16

    # ── Prefix caching ──
    enable_prefix_caching: bool = True
    prefix_caching_hash_algo: str = "xxhash"  # faster than sha256

    # ── Performance mode ──
    performance_mode: Literal["balanced", "interactivity", "throughput"] = "balanced"

    # ── Compilation ──
    optimization_level: Literal["O0", "O1", "O2", "O3"] = "O2"

    # ── Scheduling ──
    enable_chunked_prefill: bool = True
    scheduler_policy: str = "fcfs"
    scheduler_delay_factor: float | None = None

    # ── Quantization ──
    quantization: str | None = None  # fp8, awq, gptq...

    # ── Speculative decoding ──
    spec_model: str | None = None
    spec_tokens: int | None = None

    # ── CUDA Graphs ──
    cudagraph_capture_sizes: str | None = None  # e.g. "1,2,4,8,16,32,64,128,256"

    # ── Attention backend ──
    attention_backend: str | None = None  # flash_attn, flashinfer

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
            scheduler_delay_factor=0.1,
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

        if self.scheduler_delay_factor is not None:
            args.append(f"--scheduler-delay-factor={self.scheduler_delay_factor}")

        if self.cudagraph_capture_sizes:
            args.append(f"-cc.cudagraph_capture_sizes=[{self.cudagraph_capture_sizes}]")

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
    "scheduler_delay_factor",

    # Tier 4 — fine-tuning
    "cudagraph_capture_sizes",
    "prefix_caching_hash_algo",
    "attention_backend",
    "enable_prefix_caching",
]
