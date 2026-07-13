#!/usr/bin/env python3
"""Analyse benchmark_result.json and produce a structured report."""

import argparse
import json
import os
import sys
from collections import Counter

from .metrics import ERSConfig, ERSScorer, final_score


def load_benchmark(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def latency_distribution(values_ms: list[float], bins: list[float] | None = None):
    """Categorise values into latency buckets."""
    if bins is None:
        bins = [10, 20, 30, 45, 60, 100, 200, 500, 1000, 2000, 5000]
    dist = {}
    for b in bins:
        dist[f"≤{b}ms"] = sum(1 for v in values_ms if v <= b)
    dist[f">{bins[-1]}ms"] = sum(1 for v in values_ms if v > bins[-1])
    return dist


def classify_request(
    r: dict,
    config: ERSConfig,
) -> str:
    """Classify a request into quality buckets."""
    if not r["success"]:
        return "failed"
    ttft = r["ttft_ms"] / 1000
    tpot = r.get("tpot_mean_ms", 0) / 1000
    if ttft <= config.f_ttft_s and tpot <= config.f_tpot_s:
        return "perfect"
    if ttft <= config.c_ttft_s and tpot <= config.c_tpot_s:
        return "acceptable"
    return "over_threshold"


def analyze_report(report: dict, output_dir: str = "."):
    """Analyse a benchmark_result.json and produce analysis."""
    config = ERSConfig(**report.get("meta", {}).get("config", {}))
    scorer = ERSScorer(config)
    summary = report["summary"]
    latency = report["latency"]
    per_req = report["per_request"]
    trace_stats = report.get("trace_statistics", {})

    # Classify each request
    classes = Counter()
    for r in per_req:
        classes[classify_request(r, config)] += 1

    ttft_ms = [r["ttft_ms"] for r in per_req if r["success"]]
    tpot_ms = [r["tpot_mean_ms"] for r in per_req if r["success"]]

    ttft_dist = latency_distribution(ttft_ms)
    tpot_dist = latency_distribution(tpot_ms, bins=[10, 15, 20, 25, 30, 35, 40, 45, 60, 100])

    # Per-quartile ERS
    scores = sorted([r["s_request"] for r in per_req])
    n = len(scores)
    if n > 0:
        q1 = scores[n // 4]
        q2 = scores[n // 2]
        q3 = scores[3 * n // 4]
    else:
        q1 = q2 = q3 = 0

    # Accuracy
    accuracy_drop = summary.get("accuracy_drop", None)
    final = summary.get("final_score", None)

    analysis = {
        "classification": dict(classes),
        "ers_quartiles": {"q1": q1, "q2": q2, "q3": q3, "min": scores[0] if scores else 0, "max": scores[-1] if scores else 0},
        "ttft_distribution": ttft_dist,
        "tpot_distribution": tpot_dist,
        "bottleneck_analysis": {
            "ttft_healthy_pct": round(sum(1 for r in per_req if r["success"] and r["ttft_ms"] <= config.f_ttft_s * 1000) / max(len(per_req), 1) * 100, 1),
            "tpot_healthy_pct": round(sum(1 for r in per_req if r["success"] and r.get("tpot_mean_ms", 0) <= config.f_tpot_s * 1000) / max(len(per_req), 1) * 100, 1),
        },
    }

    # Markdown report
    md = []
    md.append("# Benchmark Analysis Report\n")
    md.append(f"**Trace:** {report['meta'].get('trace', 'N/A')}  ")
    md.append(f"**Model:** {report['meta'].get('model', 'N/A')}  ")
    md.append(f"**Timestamp:** {report['meta'].get('timestamp', 'N/A')}  \n")

    md.append("## Summary\n")
    md.append(f"| Metric | Value |")
    md.append(f"|---|---|")
    md.append(f"| Total requests | {summary['total_requests']} |")
    md.append(f"| Successful | {summary['successful']} |")
    md.append(f"| Failed | {summary['failed']} |")
    md.append(f"| Wall time | {summary['wall_time_s']:.1f}s |")
    md.append(f"| Throughput | {summary['throughput_req_per_s']:.2f} req/s |")
    md.append(f"| **ERS** | **{summary['overall_ers']:.6f}** |")
    if final is not None:
        md.append(f"| Accuracy drop | {accuracy_drop:.4f} |")
        md.append(f"| **Final score (100×ERS×f(Δ))** | **{final:.4f}** |")
    md.append("")

    md.append("## Latency\n")
    md.append("| Metric | Mean | Median | P95 | Min | Max |")
    md.append("|---|---|---|---|---|---|")
    md.append(f"| TTFT (ms) | {latency['ttft_ms']['mean']:.1f} | {latency['ttft_ms']['median']:.1f} | {latency['ttft_ms']['p95']:.1f} | {latency['ttft_ms']['min']:.1f} | {latency['ttft_ms']['max']:.1f} |")
    md.append(f"| TPOT (ms) | {latency['tpot_mean_ms']['mean']:.1f} | {latency['tpot_mean_ms']['median']:.1f} | {latency['tpot_mean_ms']['p95']:.1f} | {latency['tpot_mean_ms']['min']:.1f} | {latency['tpot_mean_ms']['max']:.1f} |")
    md.append("")

    md.append("## Request Classification\n")
    md.append(f"| Category | Count | % |")
    md.append(f"|---|---|---|")
    total = len(per_req)
    for cat, cnt in sorted(classes.items()):
        md.append(f"| {cat} | {cnt} | {cnt/total*100:.1f}% |")
    md.append("")

    md.append("## Latency Distribution\n")
    md.append("### TTFT\n")
    floor_ms = config.f_ttft_s * 1000
    ceil_ms = config.c_ttft_s * 1000
    md.append(f"| Bucket | Count |")
    md.append(f"|---|---|")
    md.append(f"| **Floor ≤ {floor_ms:.0f}ms** | {ttft_dist.get(f'≤{floor_ms:.0f}ms', 0)} |" if f"≤{floor_ms:.0f}ms" in ttft_dist else f"| ≤{floor_ms:.0f}ms | {ttft_dist.get(f'≤{floor_ms:.0f}ms', 0)} |")
    for k, v in sorted(ttft_dist.items()):
        md.append(f"| {k} | {v} |")

    md.append("### TPOT\n")
    md.append(f"| Bucket | Count |")
    md.append(f"|---|---|")
    for k, v in sorted(tpot_dist.items()):
        md.append(f"| {k} | {v} |")
    md.append("")

    md.append("## Bottleneck Analysis\n")
    ba = analysis["bottleneck_analysis"]
    md.append(f"- **TTFT healthy** (≤{config.f_ttft_s*1000:.0f}ms): {ba['ttft_healthy_pct']}% of requests")
    md.append(f"- **TPOT healthy** (≤{config.f_tpot_s*1000:.0f}ms): {ba['tpot_healthy_pct']}% of requests")
    if ba['tpot_healthy_pct'] < 90:
        md.append("- **⚠ TPOT is the bottleneck** — most requests exceed the 20ms floor")
    if ba['ttft_healthy_pct'] < 90:
        md.append("- **⚠ TTFT may be a bottleneck** — prefill may be too slow")
    md.append("")

    md.append("## Trace Statistics\n")
    for k, v in sorted(trace_stats.items()):
        if isinstance(v, dict):
            md.append(f"- **{k}**: mean={v.get('mean', '?'):.2f} max={v.get('max', '?'):.2f} min={v.get('min', '?'):.2f}")
        else:
            md.append(f"- **{k}**: {v}")
    md.append("")

    md_path = os.path.join(output_dir, "benchmark_analysis.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"[analyze] analysis saved to {md_path}")

    return analysis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True, help="benchmark_result.json")
    p.add_argument("--output-dir", default=".")
    args = p.parse_args()

    report = load_benchmark(args.result)
    analyze_report(report, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
