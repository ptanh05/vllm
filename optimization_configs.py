"""
Optimization Configuration Reference for AI RACE Phase 1
======================================================
Central reference for all vLLM flags optimized for Qwen3.5-2B on MiG H200 (18GB VRAM).

Model specs:
  - hidden_size=4096, num_layers=32 (8 full-attn + 24 linear-attn)
  - num_kv_heads=4, head_dim=256
  - KV cache per token (BF16) for 8 full-attn layers: 16,384 elements
  - KV cache full 256K context (BF16): ~8 GB → FP8: ~4 GB
  - Weights (BF16): ~4 GB → FP8: ~2 GB
  - VRAM available: 18 GB

Key observation: only 25% layers are full-attention with traditional KV cache.
"""

from dataclasses import dataclass, field


@dataclass
class ServingConfig:
    """Core serving configuration for optimized deployment."""
    # Model
    model: str = "/model"
    served_model_name: str = "Qwen3.5-2B"
    host: str = "0.0.0.0"
    port: int = 8000

    # Memory & context
    max_model_len: int = 262144
    gpu_memory_utilization: float = 0.95        # Leave 5% headroom

    # === QUANTIZATION (Key Optimizations) ===
    kv_cache_dtype: str = "fp8_e4m3"            # KV cache: BF16 → FP8 (save ~4GB)
    quantization: str = "fp8"                   # Weights: BF16 → FP8 (save ~2GB)

    # === SCHEDULING / BATCHING ===
    enable_prefix_caching: bool = True          # Reuse KV cache for shared prefixes
    max_num_seqs: int = 1024                    # Default=256, increased for throughput
    max_num_batched_tokens: int = 8192          # Default=2048 in API server
    # NOTE: performance_mode=throughput only doubles batching params if they
    # are NOT explicitly set. Since we set them above, explicitly, keep
    # performance_mode="balanced" (no-op) to avoid confusion.
    performance_mode: str = "balanced"          # batching set explicitly above

    # === CUDA GRAPH ===
    optimization_level: str = "O3"              # O2=default, O3=more aggressive
    # enforce_eager: bool = False               # False = use CUDA graphs

    # === BLOCK MANAGEMENT ===
    block_size: int = 32                        # Default=16, larger=less fragmentation
    use_v2_block_manager: bool = True           # Default=True in v0.22

    # === ATTENTION ===
    # attention_backend: str = ""               # auto-detect (FlashAttn or FlashInfer)

    # === ADVANCED ===
    # scheduler_delay_factor: float = 0.0      # Prefer immediate scheduling
    # num_scheduler_steps: int = 1              # Default=1

    def to_command_args(self) -> list[str]:
        """Convert config to CLI argument list for docker-compose command."""
        args = [
            f"--model={self.model}",
            f"--served-model-name={self.served_model_name}",
            f"--host={self.host}",
            f"--port={self.port}",
            f"--max-model-len={self.max_model_len}",
            f"--gpu-memory-utilization={self.gpu_memory_utilization}",
            f"--kv-cache-dtype={self.kv_cache_dtype}",
            f"--quantization={self.quantization}",
            f"--enable-prefix-caching",
            f"--max-num-seqs={self.max_num_seqs}",
            f"--max-num-batched-tokens={self.max_num_batched_tokens}",
            f"--performance-mode={self.performance_mode}",
            f"--optimization-level={self.optimization_level}",
            f"--block-size={self.block_size}",
        ]
        return args


# Pre-defined configuration presets
CONFIG_PRESETS = {
    "max_throughput": ServingConfig(
        max_num_seqs=2048,
        max_num_batched_tokens=16384,
        performance_mode="balanced",     # batching set explicitly
        optimization_level="O3",
    ),
    "balanced": ServingConfig(
        max_num_seqs=1024,
        max_num_batched_tokens=8192,
        performance_mode="balanced",     # batching set explicitly
    ),
    "low_latency": ServingConfig(
        max_num_seqs=256,                # Smaller batches = lower TPOT
        max_num_batched_tokens=4096,
        performance_mode="balanced",     # batching set explicitly
        optimization_level="O3",
    ),
    "quant_only": ServingConfig(
        # Only quantization, conservative batching
        max_num_seqs=256,
        max_num_batched_tokens=4096,
        performance_mode=None,     # type: ignore[arg-type]
        optimization_level="O2",
    ),
}
