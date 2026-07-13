#!/usr/bin/env python3
"""Phase 4: Profile vLLM serving stack — identify bottlenecks.

Collects:
  - CPU usage (per-core)
  - GPU memory, utilisation, temperature
  - Scheduler metrics (queue depth, batch size)
  - KV cache utilisation
  - CUDA graph statistics
  - Request scheduling timeline

Usage:
    # Profile against a running server
    python -m benchmark.profile --trace trace-round1.jsonl \\
        --endpoint http://localhost:8000/v1 --model Qwen3.5-2B

    # Profile with NVTX ranges (requires nsys)
    NSYS=1 python -m benchmark.profile --trace trace-round1.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

from .parser import load_trace, trace_statistics
from .client import stream_completion


def _log(msg: str):
    print(f"[profile] {msg}", flush=True)


# ── GPU monitoring ──────────────────────────────────────────────

async def sample_gpu_metrics(duration_s: float, interval_s: float = 1.0) -> list[dict]:
    """Periodically sample nvidia-smi metrics."""
    samples = []
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    samples.append({
                        "time_s": round(time.monotonic(), 2),
                        "gpu_idx": int(parts[0]),
                        "gpu_util_pct": float(parts[1]),
                        "mem_used_mib": float(parts[2]),
                        "mem_total_mib": float(parts[3]),
                        "temp_c": float(parts[4]),
                        "power_w": float(parts[5]),
                    })
        except Exception as e:
            _log(f"nvidia-smi failed: {e}")
        await asyncio.sleep(interval_s)
    return samples


# ── vLLM metrics endpoint ───────────────────────────────────────

async def fetch_vllm_metrics(endpoint: str) -> dict:
    """Fetch /metrics from vLLM's Prometheus endpoint."""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{endpoint}/metrics", timeout=10.0)
            text = resp.text

            metrics = {}
            # Parse key counters
            patterns = {
                "vllm:num_requests_running": r"vllm:num_requests_running\s+([\d.]+)",
                "vllm:num_requests_waiting": r"vllm:num_requests_waiting\s+([\d.]+)",
                "vllm:num_requests_swapped": r"vllm:num_requests_swapped\s+([\d.]+)",
                "vllm:gpu_cache_usage": r"vllm:gpu_cache_usage\s+([\d.]+)",
                "vllm:cpu_cache_usage": r"vllm:cpu_cache_usage\s+([\d.]+)",
                "vllm:prompt_tokens_total": r"vllm:prompt_tokens_total\s+([\d.]+)",
                "vllm:generation_tokens_total": r"vllm:generation_tokens_total\s+([\d.]+)",
                "vllm:time_to_first_token_seconds": r"vllm:time_to_first_token_seconds_sum\s+([\d.]+)",
                "vllm:time_per_output_token_seconds": r"vllm:time_per_output_token_seconds_sum\s+([\d.]+)",
                "vllm:request_success_total": r"vllm:request_success_total\s+([\d.]+)",
            }
            for name, pat in patterns.items():
                m = re.search(pat, text)
                if m:
                    metrics[name] = float(m.group(1))
            return metrics
        except Exception as e:
            return {"error": str(e)}
    except Exception:
        return {}


# ── VLLM LOG parsing ────────────────────────────────────────────

def parse_vllm_logs(container_name: str = "vllm-model") -> dict:
    """Extract configuration and diagnostics from vLLM server logs."""
    info = {}
    try:
        result = subprocess.run(
            ["docker", "logs", container_name],
            capture_output=True, text=True, timeout=10,
        )
        log_text = result.stdout + result.stderr

        # KV cache allocation
        m = re.search(r"Allocate blocks:\s*(\d+)\s*GPU\s*blocks", log_text)
        if m:
            info["num_gpu_blocks"] = int(m.group(1))

        m = re.search(r"# GPU blocks:\s*(\d+)", log_text)
        if m:
            info["num_gpu_blocks"] = int(m.group(1))

        # Block size
        m = re.search(r"block_size\s*=\s*(\d+)", log_text)
        if m:
            info["block_size"] = int(m.group(1))

        # Model info
        m = re.search(r"Loading model\s+(.+)", log_text)
        if m:
            info["model_loaded"] = m.group(1).strip()

        # CUDA graphs
        m = re.search(r"(enabled|disabled).*(cudagraph|cuda graph)", log_text, re.IGNORECASE)
        if m:
            info["cuda_graphs"] = m.group(1)

        # Prefix caching
        if re.search(r"prefix.*cache", log_text, re.IGNORECASE):
            info["prefix_caching"] = "enabled"

        # Warnings / errors
        warnings = re.findall(r"(WARNING|ERROR)\s+(.+)", log_text)
        info["warnings"] = [w[1] for w in warnings[:10]]

    except Exception as e:
        info["log_error"] = str(e)

    return info


# ── Profile main ────────────────────────────────────────────────

async def profile(
    trace_path: str,
    endpoint: str,
    model: str,
    output_dir: str,
    concurrency: int = 4,
    duration_s: int = 120,
    enable_gpu_monitoring: bool = True,
):
    """Profile the serving stack while replaying the trace."""
    os.makedirs(output_dir, exist_ok=True)
    requests = load_trace(trace_path)
    _log(f"loaded {len(requests)} requests from trace")

    # ── Phase 1: Static analysis ──
    _log("Phase 1: Static configuration analysis")
    static_info = {}

    # Fetch vLLM config
    metrics_before = await fetch_vllm_metrics(endpoint)
    static_info["vllm_metrics_before"] = metrics_before

    # Parse container logs
    container_logs = parse_vllm_logs()
    static_info["container_logs"] = container_logs
    _log(f"  KV cache blocks: {container_logs.get('num_gpu_blocks', 'N/A')}")
    _log(f"  CUDA graphs: {container_logs.get('cuda_graphs', 'N/A')}")
    _log(f"  prefix caching: {container_logs.get('prefix_caching', 'N/A')}")

    # ── Phase 2: Background GPU monitoring ──
    _log("Phase 2: GPU monitoring during trace replay")
    gpu_task = None
    if enable_gpu_monitoring:
        gpu_task = asyncio.create_task(
            sample_gpu_metrics(duration_s=duration_s, interval_s=1.0)
        )

    # ── Phase 3: Replay trace ──
    _log("Phase 3: Replaying trace")
    import httpx
    sem = asyncio.Semaphore(concurrency)
    raw_results = []
    t0 = time.monotonic()

    async def process_one(req):
        async with sem:
            async with httpx.AsyncClient() as client:
                timing = await stream_completion(
                    client, endpoint, model,
                    req.messages, req.max_tokens, req.temperature,
                )
                timing.request_id = req.request_id
                raw_results.append(timing)
                status = "✓" if timing.success else "✗"
                _log(f"  [{len(raw_results)}/{len(requests)}] {status} "
                     f"{req.request_id} TTFT={timing.ttft_s*1000:.0f}ms")

    for i, req in enumerate(requests):
        await process_one(req)

    wall_s = time.monotonic() - t0
    _log(f"Trace replay done in {wall_s:.1f}s")

    # ── Phase 4: Collect post-benchmark metrics ──
    _log("Phase 4: Post-benchmark metrics")
    metrics_after = await fetch_vllm_metrics(endpoint)

    gpu_samples = []
    if gpu_task:
        gpu_samples = await gpu_task
        _log(f"  Collected {len(gpu_samples)} GPU samples")

    # ── Phase 5: Analysis ──
    _log("Phase 5: Bottleneck analysis")

    gpu_utils = [s["gpu_util_pct"] for s in gpu_samples] if gpu_samples else []
    mem_used = [s["mem_used_mib"] for s in gpu_samples] if gpu_samples else []

    successful = [r for r in raw_results if r.success]
    ttft_vals = [r.ttft_s * 1000 for r in successful]
    tpot_vals = [r.tpot_mean_s * 1000 for r in successful]

    # Detect bottlenecks
    bottlenecks = []

    # GPU utilisation
    if gpu_utils:
        avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
        if avg_gpu_util < 50:
            bottlenecks.append({
                "type": "gpu_underutilized",
                "detail": f"Avg GPU util: {avg_gpu_util:.0f}% — CPU-bound or scheduler overhead",
                "severity": "high",
            })
        elif avg_gpu_util > 95:
            bottlenecks.append({
                "type": "gpu_saturated",
                "detail": f"Avg GPU util: {avg_gpu_util:.0f}% — GPU fully utilised",
                "severity": "info",
            })

    # TPOT bottlenecks
    tpot_healthy = sum(1 for v in tpot_vals if v <= 20)
    if tpot_vals and tpot_healthy / len(tpot_vals) < 0.8:
        bottlenecks.append({
            "type": "tpot_degraded",
            "detail": f"TPOT > 20ms for {(1 - tpot_healthy/len(tpot_vals))*100:.0f}% of requests",
            "severity": "critical",
        })

    # Memory pressure
    if mem_used:
        peak_mem = max(mem_used)
        total_mem = gpu_samples[0]["mem_total_mib"] if gpu_samples else 18000
        if peak_mem / total_mem > 0.95:
            bottlenecks.append({
                "type": "memory_pressure",
                "detail": f"Peak VRAM: {peak_mem:.0f}/{total_mem:.0f} MiB ({peak_mem/total_mem*100:.0f}%)",
                "severity": "critical",
            })

    # TTFT
    ttft_healthy = sum(1 for v in ttft_vals if v <= 100)
    if ttft_vals and ttft_healthy / len(ttft_vals) < 0.9:
        bottlenecks.append({
            "type": "ttft_degraded",
            "detail": f"TTFT > 100ms for {(1 - ttft_healthy/len(ttft_vals))*100:.0f}% of requests",
            "severity": "medium",
        })

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "meta": {
            "trace": os.path.basename(trace_path),
            "endpoint": endpoint,
            "model": model,
            "concurrency": concurrency,
            "wall_time_s": round(wall_s, 2),
        },
        "static_config": static_info,
        "gpu_metrics": {
            "num_samples": len(gpu_samples),
            "gpu_util_pct": {
                "mean": round(sum(gpu_utils) / len(gpu_utils), 1) if gpu_utils else 0,
                "max": round(max(gpu_utils), 1) if gpu_utils else 0,
                "min": round(min(gpu_utils), 1) if gpu_utils else 0,
            },
            "memory_mib": {
                "mean_used": round(sum(mem_used) / len(mem_used), 1) if mem_used else 0,
                "peak_used": round(max(mem_used), 1) if mem_used else 0,
                "total": gpu_samples[0]["mem_total_mib"] if gpu_samples else 0,
            },
        },
        "latency": {
            "num_successful": len(successful),
            "num_failed": len(raw_results) - len(successful),
            "ttft_ms": {
                "mean": round(sum(ttft_vals) / len(ttft_vals), 2) if ttft_vals else 0,
                "min": round(min(ttft_vals), 2) if ttft_vals else 0,
                "max": round(max(ttft_vals), 2) if ttft_vals else 0,
            },
            "tpot_ms": {
                "mean": round(sum(tpot_vals) / len(tpot_vals), 2) if tpot_vals else 0,
                "min": round(min(tpot_vals), 2) if tpot_vals else 0,
                "max": round(max(tpot_vals), 2) if tpot_vals else 0,
            },
        },
        "bottlenecks": bottlenecks,
        "recommendations": _generate_recommendations(bottlenecks, static_info),
    }

    # Save
    out_path = os.path.join(output_dir, "profiling_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    _log(f"profiling report → {out_path}")

    # Markdown report
    _save_markdown(report, output_dir)

    return report


def _generate_recommendations(bottlenecks: list[dict], static: dict) -> list[str]:
    """Generate actionable recommendations from profiling data."""
    recs = []
    log_info = static.get("container_logs", {})

    for b in bottlenecks:
        if b["type"] == "gpu_underutilized":
            recs.append("Increase --max-num-seqs and --max-num-batched-tokens to keep GPU busy")
            recs.append("Set --performance-mode=throughput for larger batch sizes")
            recs.append("If CPU-bound: reduce scheduler overhead with larger --block-size (32 or 64)")
        if b["type"] == "memory_pressure":
            recs.append("Enable --kv-cache-dtype=fp8_e4m3 to halve KV cache memory")
            recs.append("Enable --quantization=fp8 to halve model weight memory")
            recs.append("Reduce --max-model-len if full 256K context is not needed by trace")
        if b["type"] == "tpot_degraded":
            recs.append("TPOT > 20ms → need faster decode: enable CUDA graphs (O3), reduce batch size")
            recs.append("Consider --block-size=32 to reduce cache fragmentation")
            recs.append("If possible: use FlashInfer attention backend for faster attention")
        if b["type"] == "ttft_degraded":
            recs.append("Enable --enable-chunked-prefill to overlap prefill with decode")
            recs.append("Set --enable-prefix-caching for shared prefix reuse")

    if not log_info.get("cuda_graphs") == "enabled":
        recs.append("Enable CUDA graphs: --optimization-level=O3")

    if not recs:
        recs.append("No critical bottlenecks detected. Consider fine-tuning batch parameters.")

    return recs


def _save_markdown(report: dict, output_dir: str):
    """Produce profiling_report.md."""
    md = []
    meta = report["meta"]
    gpu = report["gpu_metrics"]
    lat = report["latency"]
    bots = report["bottlenecks"]

    md.append("# Profiling Report\n")
    md.append(f"**Model:** {meta['model']}  ")
    md.append(f"**Endpoint:** {meta['endpoint']}  ")
    md.append(f"**Trace:** {meta['trace']}  ")
    md.append(f"**Timestamp:** {report['timestamp']}  ")
    md.append(f"**Wall time:** {meta['wall_time_s']:.1f}s  \n")

    md.append("## GPU Metrics\n")
    md.append(f"| Metric | Value |")
    md.append(f"|---|---|")
    md.append(f"| GPU Util (mean) | {gpu['gpu_util_pct']['mean']:.1f}% |")
    md.append(f"| GPU Util (peak) | {gpu['gpu_util_pct']['max']:.1f}% |")
    md.append(f"| VRAM Used (mean) | {gpu['memory_mib']['mean_used']:.0f} MiB |")
    md.append(f"| VRAM Peak | {gpu['memory_mib']['peak_used']:.0f} MiB |")
    md.append(f"| VRAM Total | {gpu['memory_mib']['total']:.0f} MiB |")
    md.append("")

    sc = report.get("static_config", {})
    logs = sc.get("container_logs", {})
    if logs.get("num_gpu_blocks"):
        md.append(f"| GPU KV cache blocks | {logs['num_gpu_blocks']} |")
    if logs.get("block_size"):
        md.append(f"| Block size | {logs['block_size']} |")
    md.append("")

    md.append("## Latency\n")
    md.append(f"| Metric | Mean | Min | Max |")
    md.append(f"|---|---|---|---|")
    md.append(f"| TTFT (ms) | {lat['ttft_ms']['mean']:.1f} | {lat['ttft_ms']['min']:.1f} | {lat['ttft_ms']['max']:.1f} |")
    md.append(f"| TPOT (ms) | {lat['tpot_ms']['mean']:.1f} | {lat['tpot_ms']['min']:.1f} | {lat['tpot_ms']['max']:.1f} |")
    md.append("")

    md.append("## Bottlenecks\n")
    if bots:
        for b in bots:
            md.append(f"- **[{b['severity'].upper()}]** {b['detail']}")
    else:
        md.append("- No significant bottlenecks detected.")
    md.append("")

    md.append("## Recommendations\n")
    for r in report.get("recommendations", []):
        md.append(f"- {r}")
    md.append("")

    path = os.path.join(output_dir, "profiling_report.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    print(f"[profile] markdown report → {path}")


def main():
    p = argparse.ArgumentParser(description="Profile vLLM serving stack")
    p.add_argument("--trace", required=True, help="trace-round1.jsonl")
    p.add_argument("--endpoint", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen3.5-2B")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--output-dir", default=".")
    p.add_argument("--duration", type=int, default=120,
                   help="GPU monitoring duration (seconds)")
    p.add_argument("--no-gpu", action="store_true",
                   help="Disable GPU monitoring (no nvidia-smi)")
    args = p.parse_args()

    asyncio.run(profile(
        trace_path=args.trace,
        endpoint=args.endpoint,
        model=args.model,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        duration_s=args.duration,
        enable_gpu_monitoring=not args.no_gpu,
    ))


if __name__ == "__main__":
    main()
