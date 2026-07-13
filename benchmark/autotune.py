#!/usr/bin/env python3
"""Phase 3: Automatic parameter search for vLLM serving.

Searches combinations of:
  gpu-memory-utilization
  block-size
  max-num-seqs
  max-num-batched-tokens
  performance-mode
  kv-cache-dtype
  optimization-level
  enable-chunked-prefill
  enable-prefix-caching
  scheduler-delay-factor

Usage:
    # Dry-run: analyse the trace and generate candidate configs
    python -m benchmark.autotune --trace trace-round1.jsonl --dry-run

    # Full search: restart docker-compose for each config, benchmark, collect ERS
    python -m benchmark.autotune --trace trace-round1.jsonl \\
        --endpoint http://localhost:8000/v1 --model Qwen3.5-2B \\
        --compose-file docker-compose.yml --max-trials 20
"""

import argparse
import asyncio
import csv
import itertools
import json
import os
import random
import subprocess
import sys
import time
import datetime
from copy import deepcopy
from typing import Any

from .parser import load_trace, trace_statistics
from .replay_trace import run_replay
from .metrics import ERSConfig, ERSScorer


# ---- Candidate space ----

CANDIDATE_SPACE: dict[str, list[Any]] = {
    "gpu-memory-utilization": [0.85, 0.90, 0.95, 0.97],
    "block-size": [16, 32, 64],
    "max-num-seqs": [256, 512, 1024, 2048],
    # max-num-batched-tokens = None means auto
    "max-num-batched-tokens": [None, 2048, 4096, 8192],
    "performance-mode": ["balanced", "throughput"],
    "kv-cache-dtype": ["auto", "fp8_e4m3"],
    "optimization-level": ["O2", "O3"],
    "enable-chunked-prefill": [None],  # None = rely on vLLM default (True)
    "enable-prefix-caching": [None],
}

# Fixed params that stay constant across trials
FIXED_PARAMS: dict[str, Any] = {
    "model": "/model",
    "served-model-name": "Qwen3.5-2B",
    "host": "0.0.0.0",
    "port": 8000,
    "max-model-len": 262144,
    "tensor-parallel-size": 1,
}


def generate_candidates(max_trials: int = 20, seed: int = 42) -> list[dict[str, Any]]:
    """Generate a diverse set of candidate configurations via random search."""
    rng = random.Random(seed)

    # Always evaluate the baseline first
    candidates = [
        {
            "gpu-memory-utilization": 0.95,
            "block-size": 16,
            "max-num-seqs": 256,
            "max-num-batched-tokens": None,
            "performance-mode": "balanced",
            "kv-cache-dtype": "auto",
            "optimization-level": "O2",
            "enable-chunked-prefill": None,
            "enable-prefix-caching": True,
        },
        # Then a sensible optimized baseline
        {
            "gpu-memory-utilization": 0.95,
            "block-size": 32,
            "max-num-seqs": 1024,
            "max-num-batched-tokens": None,
            "performance-mode": "throughput",
            "kv-cache-dtype": "fp8_e4m3",
            "optimization-level": "O3",
            "enable-chunked-prefill": None,
            "enable-prefix-caching": True,
        },
    ]

    keys = list(CANDIDATE_SPACE.keys())
    remaining = max_trials - len(candidates)
    for _ in range(remaining):
        cand = {}
        for k in keys:
            vals = CANDIDATE_SPACE[k]
            cand[k] = vals[rng.randint(0, len(vals) - 1)]
        # Set enable-chunked-prefill and enable-prefix-caching to sensical values
        if cand.get("enable-chunked-prefill") is None:
            cand["enable-chunked-prefill"] = True if rng.random() > 0.3 else False
        if cand.get("enable-prefix-caching") is None:
            cand["enable-prefix-caching"] = True if rng.random() > 0.2 else False
        candidates.append(cand)

    return candidates


def build_docker_compose(candidate: dict[str, Any], compose_template_path: str) -> str:
    """Generate a docker-compose.yml from a candidate config and the template."""
    try:
        import yaml
    except ImportError:
        yaml = None  # fallback: manual YAML generation

    with open(compose_template_path) as f:
        template = f.read()

    # Build command list
    cmd = []
    for param_key, val in candidate.items():
        if val is None:
            continue
        flag = f"--{param_key}"
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        else:
            cmd.append(flag)
            cmd.append(str(val))

    # Actually we need a more structured approach. Let's just directly build YAML.
    compose = {
        "services": {
            "model": {
                "image": "vllm/vllm-openai:v0.22.1",
                "entrypoint": [
                    "python3",
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                ],
                "command": [
                    "--model=/model",
                    "--served-model-name=Qwen3.5-2B",
                    "--host=0.0.0.0",
                    "--port=8000",
                    "--max-model-len=262144",
                    "--tensor-parallel-size=1",
                ],
                "ports": ["8000:8000"],
                "shm_size": "2g",
                "deploy": {
                    "resources": {
                        "reservations": {
                            "devices": [
                                {
                                    "driver": "nvidia",
                                    "count": 1,
                                    "capabilities": ["gpu"],
                                }
                            ]
                        }
                    }
                },
            }
        }
    }

    # Add candidate flags
    for key, val in candidate.items():
        if val is None:
            continue
        rf = f"--{key.replace('_', '-')}"
        if isinstance(val, bool):
            if val:
                compose["services"]["model"]["command"].append(rf)
        else:
            compose["services"]["model"]["command"].append(rf)
            compose["services"]["model"]["command"].append(str(val))

    if yaml:
        return yaml.dump(compose, default_flow_style=False)
    else:
        # Simple fallback: write as JSON (Docker Compose v3 accepts JSON format)
        import json as _json
        return _json.dumps(compose, indent=2)


async def run_trial(
    candidate: dict[str, Any],
    idx: int,
    trace_path: str,
    endpoint: str,
    model: str,
    concurrency: int,
    output_dir: str,
    compose_template: str | None = None,
) -> dict[str, Any]:
    """Run a single trial: (optionally restart compose → wait → benchmark → score)."""
    print("\n" + "=" * 70)
    print(f"  TRIAL {idx}")
    print("=" * 70)
    for k, v in candidate.items():
        print(f"    {k}: {v}")

    if compose_template:
        # Generate and write compose file, restart
        compose_yml = build_docker_compose(candidate, compose_template)
        compose_path = os.path.join(output_dir, f"compose_trial_{idx:03d}.yml")
        with open(compose_path, "w") as f:
            f.write(compose_yml)
        print(f"  [trial {idx}] wrote {compose_path}")

        # Stop existing
        subprocess.run(
            ["docker", "compose", "-f", compose_path, "down", "--timeout", "10"],
            capture_output=True, timeout=60,
        )

        # Start new
        subprocess.run(
            ["docker", "compose", "-f", compose_path, "up", "-d"],
            capture_output=True, timeout=300,
        )
        print(f"  [trial {idx}] container started, waiting for model to load...")
        # Wait for health
        await _wait_for_health(endpoint, timeout_s=300)
        print(f"  [trial {idx}] health OK")

    # Warmup (short request)
    print(f"  [trial {idx}] warmup...")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            await client.post(
                f"{endpoint}/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
                timeout=60.0,
            )
    except Exception as e:
        print(f"  [trial {idx}] warmup failed: {e}")

    # Benchmark with trace
    trial_dir = os.path.join(output_dir, f"trial_{idx:03d}")
    os.makedirs(trial_dir, exist_ok=True)
    try:
        report = await run_replay(
            trace_path=trace_path,
            endpoint=endpoint,
            model=model,
            concurrency=concurrency,
            output_dir=trial_dir,
            respect_arrivals=True,
        )
        ers = report["summary"]["overall_ers"]
        successful = report["summary"]["successful"]
        total = report["summary"]["total_requests"]
        mean_tpot = report["latency"]["tpot_mean_ms"]["mean"]
        mean_ttft = report["latency"]["ttft_ms"]["mean"]
        print(f"  [trial {idx}] ERS={ers:.6f}  successful={successful}/{total}  "
              f"TTFT={mean_ttft:.1f}ms  TPOT={mean_tpot:.1f}ms")
    except Exception as e:
        print(f"  [trial {idx}] BENCHMARK FAILED: {e}")
        report = None
        ers = 0.0
        successful = 0
        mean_tpot = 999
        mean_ttft = 999

    return {
        "trial": idx,
        "config": candidate,
        "ers": round(ers, 6),
        "successful": successful,
        "total_requests": total if 'total' in dir() else 0,
        "mean_ttft_ms": round(mean_ttft, 2),
        "mean_tpot_ms": round(mean_tpot, 2),
        "report_path": str(trial_dir / "benchmark_result.json") if report else None,
    }


async def _wait_for_health(endpoint: str, timeout_s: int = 300):
    import httpx
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(f"{endpoint}/health", timeout=5.0)
                if r.status_code == 200:
                    return
        except Exception:
            pass
        await asyncio.sleep(5)
    raise RuntimeError(f"Server not healthy after {timeout_s}s")


async def main():
    p = argparse.ArgumentParser(description="vLLM Automatic Parameter Search")
    p.add_argument("--trace", required=True, help="trace-round1.jsonl")
    p.add_argument("--endpoint", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen3.5-2B")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output-dir", default="tune_results")
    p.add_argument("--compose-file", default=None,
                   help="docker-compose.yml template (omit for endpoint-only mode)")
    p.add_argument("--max-trials", type=int, default=20)
    p.add_argument("--dry-run", action="store_true",
                   help="Just generate candidate configs, don't run benchmarks")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    candidates = generate_candidates(max_trials=args.max_trials, seed=args.seed)
    print(f"[autotune] generated {len(candidates)} candidate configs")

    # Save candidate configs
    cand_path = os.path.join(args.output_dir, "candidates.json")
    with open(cand_path, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"[autotune] candidates saved to {cand_path}")

    if args.dry_run:
        print("\nCandidate configs:")
        for i, c in enumerate(candidates):
            print(f"  [{i}] {json.dumps(c)}")
        sys.exit(0)

    # Run trials sequentially (each restarts the container)
    results: list[dict] = []
    for i, cand in enumerate(candidates):
        result = await run_trial(
            cand, i,
            trace_path=args.trace,
            endpoint=args.endpoint,
            model=args.model,
            concurrency=args.concurrency,
            output_dir=args.output_dir,
            compose_template=args.compose_file,
        )
        results.append(result)

        # Save leaderboard after every trial
        results.sort(key=lambda r: r["ers"], reverse=True)
        lb_path = os.path.join(args.output_dir, "leaderboard.csv")
        with open(lb_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "trial", "ers", "successful", "total_requests",
                "mean_ttft_ms", "mean_tpot_ms", "config",
            ])
            writer.writeheader()
            for r in results:
                row = {k: r[k] for k in writer.fieldnames if k != "config"}
                row["config"] = json.dumps(r.get("config", {}))
                writer.writerow(row)

        # Print current leaderboard
        print(f"\n{'='*70}")
        print(f"  LEADERBOARD (after trial {i+1}/{len(candidates)})")
        print(f"{'='*70}")
        print(f"  {'Rank':>4} {'ERS':>8} {'Success':>7} {'TTFT':>7} {'TPOT':>7}  Config")
        print(f"  {'-'*4} {'-'*8} {'-'*7} {'-'*7} {'-'*7}  {'-'*30}")
        for rank, r in enumerate(results[:10]):
            cfg = json.dumps({k: v for k, v in r.get("config", {}).items() if v is not None and v is not False})
            print(f"  {rank+1:>4} {r['ers']:>8.4f} {r['successful']:>3}/{r['total_requests']:>3} "
                  f"{r['mean_ttft_ms']:>7.1f} {r['mean_tpot_ms']:>7.1f}  {cfg[:60]}")

        # Save best config
        if results:
            best = results[0]
            best_path = os.path.join(args.output_dir, "best_config.json")
            with open(best_path, "w") as f:
                json.dump(best, f, indent=2)

    print(f"\n[autotune] complete. Leaderboard: {lb_path}")
    print(f"[autotune] Best config: {best_path}")


if __name__ == "__main__":
    asyncio.run(main())
