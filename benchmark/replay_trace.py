#!/usr/bin/env python3
"""Replay trace-round1.jsonl against an OpenAI-compatible endpoint.

Usage:
    python -m benchmark.replay_trace --trace trace-round1.jsonl \\
        --endpoint http://localhost:8000/v1 --model Qwen3.5-2B
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime

import httpx

from .parser import load_trace, trace_statistics
from .client import stream_completion
from .metrics import ERSScorer, ERSConfig, final_score


async def run_replay(
    trace_path: str,
    endpoint: str,
    model: str,
    concurrency: int = 8,
    output_dir: str = ".",
    respect_arrivals: bool = True,
) -> dict:
    """Replay the trace and return the full report dict."""
    requests = load_trace(trace_path)
    print(f"[replay] loaded {len(requests)} requests from {trace_path}")

    stats = trace_statistics(requests)
    print(f"[replay] trace duration: {stats['total_duration_s']:.1f}s, "
          f"arrival rate: {stats['arrival_rate_req_per_s']:.2f} req/s")

    scorer = ERSScorer(ERSConfig())
    sem = asyncio.Semaphore(concurrency)

    raw_timings = []
    t0 = time.monotonic()

    async def process_one(req, idx) -> dict:
        async with sem:
            rid = req.request_id
            print(f"[replay] [{idx+1}/{len(requests)}] {rid} — sending...")

            # Respect trace-relative arrival times
            if respect_arrivals and idx > 0:
                prev_ts = requests[idx - 1].arrival_time
                gap = req.arrival_time - prev_ts
                if gap > 0.005:  # >=5ms delay
                    await asyncio.sleep(gap)

            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
                timing = await stream_completion(
                    client,
                    endpoint,
                    model,
                    req.messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                )
            timing.request_id = rid

            score = scorer.score_one(timing)
            ttft_ms = timing.ttft_s * 1000
            tpot_ms = timing.tpot_mean_s * 1000

            status = "✓" if timing.success else "✗"
            print(f"[replay] [{idx+1}/{len(requests)}] {status} {rid} — "
                  f"TTFT={ttft_ms:.1f}ms TPOT={tpot_ms:.1f}ms "
                  f"tok={timing.num_output_tokens} "
                  f"score={['FAILED','OK'][timing.success]}")

            rec = {
                "request_id": rid,
                "arrival_time": req.arrival_time,
                "success": timing.success,
                "ttft_ms": round(ttft_ms, 2),
                "tpot_mean_ms": round(tpot_ms, 2),
                "num_output_tokens": timing.num_output_tokens,
                "num_prompt_tokens": req.prompt_length,
                "s_ttft": round(score.s_ttft, 6),
                "s_tpot": round(score.s_tpot, 6),
                "s_request": round(score.s_request, 6),
                "total_latency_s": round(timing.total_latency_s, 3),
                "error": timing.error if not timing.success else None,
            }
            raw_timings.append(timing)
            return rec

    tasks = [process_one(req, i) for i, req in enumerate(requests)]
    per_request = await asyncio.gather(*tasks)

    wall_s = time.monotonic() - t0

    scores = [scorer.score_one(t) for t in raw_timings]
    ers = scorer.compute_ers(scores)
    successful = [t for t in raw_timings if t.success]

    ttft_vals = [t.ttft_s * 1000 for t in successful]
    tpot_vals = [t.tpot_mean_s * 1000 for t in successful]

    report = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat(),
            "trace": os.path.basename(trace_path),
            "endpoint": endpoint,
            "model": model,
            "concurrency": concurrency,
            "config": scorer.config.to_dict(),
        },
        "trace_statistics": stats,
        "summary": {
            "total_requests": len(requests),
            "successful": len(successful),
            "failed": len(requests) - len(successful),
            "wall_time_s": round(wall_s, 2),
            "throughput_req_per_s": round(len(requests) / wall_s, 2) if wall_s > 0 else 0,
            "overall_ers": round(ers, 6),
            "overall_ers_pct": round(ers * 100, 4),
        },
        "latency": {
            "ttft_ms": {
                "mean": round(sum(ttft_vals) / len(ttft_vals), 2) if ttft_vals else 0,
                "median": round(sorted(ttft_vals)[len(ttft_vals)//2], 2) if ttft_vals else 0,
                "p95": round(sorted(ttft_vals)[int(len(ttft_vals) * 0.95)], 2) if len(ttft_vals) >= 20 else 0,
                "min": round(min(ttft_vals), 2) if ttft_vals else 0,
                "max": round(max(ttft_vals), 2) if ttft_vals else 0,
            },
            "tpot_mean_ms": {
                "mean": round(sum(tpot_vals) / len(tpot_vals), 2) if tpot_vals else 0,
                "median": round(sorted(tpot_vals)[len(tpot_vals)//2], 2) if tpot_vals else 0,
                "p95": round(sorted(tpot_vals)[int(len(tpot_vals) * 0.95)], 2) if len(tpot_vals) >= 20 else 0,
                "min": round(min(tpot_vals), 2) if tpot_vals else 0,
                "max": round(max(tpot_vals), 2) if tpot_vals else 0,
            },
        },
        "per_request": per_request,
    }

    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "benchmark_result.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[replay] results saved to {out_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, "benchmark_result.csv")
    _save_csv(csv_path, per_request)
    print(f"[replay] CSV saved to {csv_path}")

    # Save summary text
    _save_summary(out_path)

    return report


def _save_csv(path: str, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _save_summary(json_path: str):
    """Print and save a human-readable summary."""
    with open(json_path) as f:
        r = json.load(f)
    s = r["summary"]
    lat = r["latency"]
    print("\n" + "=" * 64)
    print("  ERS BENCHMARK RESULT")
    print("=" * 64)
    print(f"  Requests:    {s['total_requests']} total, {s['successful']} ok, {s['failed']} failed")
    print(f"  Wall time:   {s['wall_time_s']:.1f}s")
    print(f"  Throughput:  {s['throughput_req_per_s']:.2f} req/s")
    print(f"  ERS:         {s['overall_ers']:.6f}  ({s['overall_ers_pct']:.2f}%)")
    print(f"  --- Latency ---")
    print(f"  TTFT (ms):   mean={lat['ttft_ms']['mean']:.1f}  "
          f"p50={lat['ttft_ms']['median']:.1f}  p95={lat['ttft_ms']['p95']:.1f}")
    print(f"  TPOT (ms):   mean={lat['tpot_mean_ms']['mean']:.1f}  "
          f"p50={lat['tpot_mean_ms']['median']:.1f}  p95={lat['tpot_mean_ms']['p95']:.1f}")

    config = r.get("meta", {}).get("config", {})
    print(f"  --- Scoring config ---")
    print(f"  TTFT F={config.get('f_ttft_s',0)*1000:.0f}ms C={config.get('c_ttft_s',0)*1000:.0f}ms")
    print(f"  TPOT F={config.get('f_tpot_s',0)*1000:.0f}ms C={config.get('c_tpot_s',0)*1000:.0f}ms")
    print(f"  gamma={config.get('gamma',2)}  w_ttft={config.get('w',0.5)}")
    print("=" * 64)
    return s


def main():
    parser = argparse.ArgumentParser(description="Replay trace against LLM endpoint")
    parser.add_argument("--trace", required=True, help="Path to trace-round1.jsonl")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3.5-2B")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--no-respect-arrivals", action="store_true",
                        help="Ignore trace arrival timing (fire all at once)")
    args = parser.parse_args()

    asyncio.run(run_replay(
        trace_path=args.trace,
        endpoint=args.endpoint,
        model=args.model,
        concurrency=args.concurrency,
        output_dir=args.output_dir,
        respect_arrivals=not args.no_respect_arrivals,
    ))


if __name__ == "__main__":
    main()
