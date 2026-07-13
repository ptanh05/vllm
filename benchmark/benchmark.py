#!/usr/bin/env python3
"""One-shot benchmark runner: health-check → replay → GPQA → report."""

import argparse
import asyncio
import json
import os
import sys
import time
import httpx

from .replay_trace import run_replay
from .metrics import final_score


async def check_health(endpoint: str, timeout_s: int = 120) -> bool:
    """Wait for the server health endpoint to return 200."""
    print(f"[health] waiting for {endpoint}/health ...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            async with httpx.AsyncClient() as c:
                resp = await c.get(f"{endpoint}/health", timeout=5.0)
                if resp.status_code == 200:
                    print(f"[health] ready in {time.monotonic()-t0:.0f}s")
                    return True
        except Exception:
            pass
        await asyncio.sleep(2)
    print(f"[health] NOT READY after {timeout_s}s")
    return False


async def run_benchmark(
    trace: str,
    endpoint: str,
    model: str,
    concurrency: int,
    output_dir: str,
    respect_arrivals: bool,
    gpqa_script: str | None = None,
) -> dict:
    """Run the full benchmark suite and return the report."""
    if not await check_health(endpoint):
        print("[benchmark] server not available — aborting")
        sys.exit(1)

    report = await run_replay(
        trace_path=trace,
        endpoint=endpoint,
        model=model,
        concurrency=concurrency,
        output_dir=output_dir,
        respect_arrivals=respect_arrivals,
    )

    # GPQA accuracy (external script, if provided)
    accuracy_drop = None
    if gpqa_script and os.path.isfile(gpqa_script):
        print(f"\n[benchmark] running GPQA accuracy test ({gpqa_script}) ...")
        import subprocess
        gpqa_out = os.path.join(output_dir, "gpqa_results.json")
        result = subprocess.run(
            ["python3", gpqa_script, "--endpoint", endpoint,
             "--model", model, "--output", gpqa_out],
            capture_output=True, text=True, timeout=600,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"[benchmark] GPQA test stderr:\n{result.stderr}")
        if os.path.isfile(gpqa_out):
            with open(gpqa_out) as f:
                g = json.load(f)
            accuracy_drop = g.get("delta", 0.0)
            report["accuracy"] = g

    ers = report["summary"]["overall_ers"]
    if accuracy_drop is not None:
        final = final_score(ers, accuracy_drop)
        report["summary"]["accuracy_drop"] = accuracy_drop
        report["summary"]["final_score"] = round(final, 4)
        print(f"\n[benchmark] Accuracy drop: {accuracy_drop:.4f}")
        print(f"[benchmark] Final score:   {final:.4f}")

    # Save combined report
    combined_path = os.path.join(output_dir, "benchmark_result.json")
    with open(combined_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[benchmark] full report → {combined_path}")

    return report


def main():
    p = argparse.ArgumentParser(description="vLLM Benchmark Suite")
    p.add_argument("--trace", required=True)
    p.add_argument("--endpoint", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen3.5-2B")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output-dir", default=".")
    p.add_argument("--no-respect-arrivals", action="store_true")
    p.add_argument("--gpqa-script", default=None, help="Path to benchmark_gpqa.py")
    args = p.parse_args()

    asyncio.run(run_benchmark(
        trace=args.trace,
        endpoint=args.endpoint,
        model=args.model,
        concurrency=args.concurrency,
        output_dir=args.output_dir,
        respect_arrivals=not args.no_respect_arrivals,
        gpqa_script=args.gpqa_script,
    ))


if __name__ == "__main__":
    main()
