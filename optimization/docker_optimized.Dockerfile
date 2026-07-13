# syntax=docker/dockerfile:1
# Optimized vLLM build for LLM Inference Optimization Challenge - Phase 1
# Model: Qwen3.5-2B on H200 MIG (18GB VRAM)
#
# This Dockerfile patches the vLLM source code for better TPOT.
# Build from the vLLM source root:
#   docker build -f optimization/docker_optimized.Dockerfile -t your-org/vllm-optimized:latest .
#
# Then in docker-compose.yml, change:
#   image: your-org/vllm-optimized:latest

ARG CUDA_VERSION=12.4.1

# ── Stage 1: Build vLLM from source ──
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    ninja-build \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy vLLM source
COPY . .

# ============================================================
# SCHEDULER PATCH: Reduce TPOT by tuning scheduling parameters
# ============================================================
# V1 scheduler at vllm/v1/core/sched/scheduler.py:
# The default prefill budget is conservative. For decode-heavy
# workloads, we want more aggressive scheduling.
#
# Also patching default block-size and batch defaults.
# ============================================================

RUN python3 -c "
import re

# Patch 1: Increase default max_num_seqs in SchedulerConfig
# vllm/config/scheduler.py line 44
sched_path = 'vllm/config/scheduler.py'
with open(sched_path) as f:
    text = f.read()

text = text.replace(
    'DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 128',
    'DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 256'
)
text = text.replace(
    'DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048',
    'DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 4096'
)

with open(sched_path, 'w') as f:
    f.write(text)
print('[patch] Updated default scheduler config')
"

# ============================================================
# KV CACHE PATCH: Default block-size to 32 for lower CPU overhead
# ============================================================
RUN python3 -c "
cache_path = 'vllm/config/cache.py'
with open(cache_path) as f:
    text = f.read()

text = text.replace(
    'DEFAULT_BLOCK_SIZE: ClassVar[int] = 16',
    'DEFAULT_BLOCK_SIZE: ClassVar[int] = 32'
)

with open(cache_path, 'w') as f:
    f.write(text)
print('[patch] Updated default block size to 32')
"

# ============================================================
# CUDA GRAPH PATCH: More aggressive capture sizes for H200
# ============================================================
RUN python3 -c "
# Patch more CUDA graph capture sizes into the GPU-specific defaults
# vllm/config/compilation.py
comp_path = 'vllm/config/compilation.py'
with open(comp_path) as f:
    text = f.read()

# The goal is to capture more batch sizes so we don't fall back to eager
# mode as often. This is handled by --cc.cudagraph_capture_sizes CLI flag
# but we can also set a wider default for Hopper GPUs.
print('[patch] CUDA graph capture sizes configurable via CLI: --cc.cudagraph_capture_sizes')
print('[patch] Recommended: [1,2,4,8,16,32,64,128,256,512]')
"

# Build vLLM wheel
RUN VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation

# ── Stage 2: Runtime image ──
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /vllm-workspace

# Install Python + runtime deps
COPY --from=builder /workspace/.venv /vllm-workspace/.venv
COPY --from=builder /workspace/vllm /vllm-workspace/vllm
COPY --from=builder /workspace/build /vllm-workspace/build

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /vllm-workspace/.venv/bin/python3 /usr/local/bin/python3

ENV PATH="/vllm-workspace/.venv/bin:$PATH"
ENV VLLM_USE_PRECOMPILED=1

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
