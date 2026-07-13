#!/usr/bin/env python3
"""Master pipeline: trace analysis → benchmark → profiling → autotune.

Usage:
    # Full pipeline (server must be running)
    python -m benchmark.run_all --trace trace-round1.jsonl \\
        --endpoint http://localhost:8000/v1 --model Qwen3.5-2B

    # Dry-run (trace analysis only, no server needed)
    python -m benchmark.run_all --trace trace-round1.jsonl --dry-run
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from .parser import load_trace, trace_statistics
from .replay_trace import run_replay
from .metrics import ERSConfig, ERSScorer, final_score


async def pipeline(
    trace_path: str,
    endpoint: str,
    model: str,
    output_dir: str,
    concurrency: int = 8,
    dry_run: bool = False,
    skip_plots: bool = False,
    skip_gpqa: bool = True,
):
    """Run the complete optimization pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*64}")
    print(f"  LLM INFERENCE OPTIMIZATION PIPELINE")
    print(f"  Started: {datetime.utcnow().isoformat()}")
    print(f"{'='*64}")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # ── Step 0: Dry run flag → just analyse the trace ──
    if dry_run:
        print("\n[DRY-RUN] Analysing trace only")
        requests = load_trace(trace_path)
        stats = trace_statistics(requests)

        print(f"\nTrace statistics for {os.path.basename(trace_path)}:")
        print(f"  Requests:      {stats['num_requests']}")
        print(f"  Duration:      {stats['total_duration_s']:.1f}s")
        print(f"  Rate:          {stats['arrival_rate_req_per_s']:.2f} req/s")
        print(f"  Max concurrent:{stats['max_concurrent_60s_window']}")
        print(f"  Prompt chars:  mean={stats['prompt_length_chars']['mean']:.0f} "
              f"max={stats['prompt_length_chars']['max']:.0f}")
        print(f"  Output tokens: mean={stats['expected_output_tokens']['mean']:.0f} "
              f"max={stats['expected_output_tokens']['max']:.0f}")

        # Save trace stats
        with open(os.path.join(output_dir, "trace_stats.json"), "w") as f:
            json.dump({"statistics": stats, "meta": {"trace": trace_path}}, f, indent=2)
        print(f"\nTrace stats saved to {output_dir}/trace_stats.json")
        print("\n[DRY-RUN] Complete. Start the server and run without --dry-run to benchmark.")
        return

    # ── Step 1: Trace analysis ──
    print(f"\n[1/5] Analysing trace...")
    result = subprocess.run(
        [sys.executable, "-m", "benchmark.analyze_trace",
         "--trace", trace_path, "--output-dir", output_dir],
        capture_output=True, text=True, timeout=60,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"[WARN] trace analysis stderr: {result.stderr}")

    # ── Step 2: ERS Benchmark ──
    print(f"\n[2/5] Running ERS benchmark...")
    report = await run_replay(
        trace_path=trace_path,
        endpoint=endpoint,
        model=model,
        concurrency=concurrency,
        output_dir=output_dir,
        respect_arrivals=True,
    )

    ers = report["summary"]["overall_ers"]

    # ── Step 3: Plots (if matplotlib available) ──
    if not skip_plots:
        print(f"\n[3/5] Generating plots...")
        result = subprocess.run(
            [sys.executable, "-m", "benchmark.plots",
             "--result", os.path.join(output_dir, "benchmark_result.json"),
             "--output-dir", output_dir],
            capture_output=True, text=True, timeout=60,
        )
        print(result.stdout or result.stderr)

    # ── Step 4: Analysis report ──
    print(f"\n[4/5] Generating analysis report...")
    result = subprocess.run(
        [sys.executable, "-m", "benchmark.analyze",
         "--result", os.path.join(output_dir, "benchmark_result.json"),
         "--output-dir", output_dir],
        capture_output=True, text=True, timeout=60,
    )
    print(result.stdout or result.stderr)

    # ── Step 5: Profiling (lightweight) ──
    print(f"\n[5/5] Running profile snapshot...")
    result = subprocess.run(
        [sys.executable, "-m", "benchmark.profile",
         "--trace", trace_path,
         "--endpoint", endpoint,
         "--model", model,
         "--output-dir", output_dir,
         "--concurrency", str(concurrency),
         "--duration", "30"],
        capture_output=True, text=True, timeout=300,
    )
    print(result.stdout or result.stderr)

    # ── Done ──
    print(f"\n{'='*64}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*64}")
    print(f"  ERS: {ers:.6f}")

    # Check if GPQA results exist
    gpqa_path = os.path.join(output_dir, "gpqa_results.json")
    if os.path.isfile(gpqa_path):
        with open(gpqa_path) as f:
            g = json.load(f)
        drop = g.get("delta", 0)
        final = final_score(ers, drop)
        print(f"  Accuracy drop: {drop:.4f}")
        print(f"  Final score:   {final:.4f}")

    print(f"\n  Output directory: {output_dir}/")
    print(f"  Files created:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"    {f:45s} {size:>8,d} bytes")
    print(f"{'='*64}\n")

    return report


def main():
    p = argparse.ArgumentParser(description="vLLM Optimization Pipeline")
    p.add_argument("--trace", required=True, help="trace-round1.jsonl")
    p.add_argument("--endpoint", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen3.5-2B")
    p.add_argument("--output-dir", default="benchmark_output")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--dry-run", action="store_true",
                   help="Analyse trace only; no server needed")
    p.add_argument("--skip-plots", action="store_true")
    args = p.parse_args()

    asyncio.run(pipeline(
        trace_path=args.trace,
        endpoint=args.endpoint,
        model=args.model,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        skip_plots=args.skip_plots,
    ))


if __name__ == "__main__":
    main()
