#!/usr/bin/env python3
"""Generate plots from benchmark results."""

import argparse
import json
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def set_style():
    if not HAS_MPL:
        return
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 120,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_latency_timeline(report: dict, output_dir: str):
    """Plot per-request TTFT and TPOT over time."""
    if not HAS_MPL:
        return

    per_req = report.get("per_request", [])
    if not per_req:
        return

    successful = [r for r in per_req if r["success"]]
    if not successful:
        return

    set_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    indices = list(range(len(successful)))
    ttft_vals = [r["ttft_ms"] for r in successful]
    tpot_vals = [r["tpot_mean_ms"] for r in successful]

    # TTFT
    ax1.bar(indices, ttft_vals, color="steelblue", alpha=0.8, width=0.8)
    ax1.axhline(100, color="green", linestyle="--", label="Floor (100ms)")
    ax1.axhline(1500, color="red", linestyle="--", label="Ceiling (1500ms)")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title("Per-Request TTFT")
    ax1.legend()
    ax1.set_yscale("log")

    # TPOT
    ax2.bar(indices, tpot_vals, color="coral", alpha=0.8, width=0.8)
    ax2.axhline(20, color="green", linestyle="--", label="Floor (20ms)")
    ax2.axhline(45, color="red", linestyle="--", label="Ceiling (45ms)")
    ax2.set_ylabel("Mean TPOT (ms)")
    ax2.set_xlabel("Request index (sorted by arrival)")
    ax2.set_title("Per-Request Mean TPOT")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "latency_timeline.png")
    plt.savefig(path)
    plt.close()
    print(f"[plots] → {path}")


def plot_latency_histogram(report: dict, output_dir: str):
    """Plot histogram of TTFT and TPOT."""
    if not HAS_MPL:
        return

    per_req = report.get("per_request", [])
    successful = [r for r in per_req if r["success"]]
    if not successful:
        return

    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ttft_vals = [r["ttft_ms"] for r in successful]
    tpot_vals = [r["tpot_mean_ms"] for r in successful]

    # Clip outliers for better visualisation
    ttft_clipped = [min(v, 2000) for v in ttft_vals]
    tpot_clipped = [min(v, 100) for v in tpot_vals]

    ax1.hist(ttft_clipped, bins=30, color="steelblue", alpha=0.8, edgecolor="white")
    ax1.axvline(100, color="green", linestyle="--", label="Floor")
    ax1.axvline(1500, color="red", linestyle="--", label="Ceiling")
    ax1.set_xlabel("TTFT (ms, capped at 2000)")
    ax1.set_ylabel("Count")
    ax1.set_title("TTFT Distribution")
    ax1.legend()

    ax2.hist(tpot_clipped, bins=30, color="coral", alpha=0.8, edgecolor="white")
    ax2.axvline(20, color="green", linestyle="--", label="Floor")
    ax2.axvline(45, color="red", linestyle="--", label="Ceiling")
    ax2.set_xlabel("Mean TPOT (ms, capped at 100)")
    ax2.set_ylabel("Count")
    ax2.set_title("TPOT Distribution")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "latency_histogram.png")
    plt.savefig(path)
    plt.close()
    print(f"[plots] → {path}")


def plot_throughput_chart(report: dict, output_dir: str):
    """Plot request timeline with cumulative throughput."""
    if not HAS_MPL:
        return

    per_req = report.get("per_request", [])
    if not per_req:
        return

    set_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    # Cumulative successful requests
    cumulative = []
    count = 0
    times = []
    for i, r in enumerate(per_req):
        if r["success"]:
            count += 1
        cumulative.append(count)
        times.append(i)

    ax.plot(times, cumulative, color="darkgreen", linewidth=2)
    ax.set_xlabel("Request index")
    ax.set_ylabel("Cumulative successful requests")
    ax.set_title("Cumulative Throughput")

    total = len(per_req)
    failed = sum(1 for r in per_req if not r["success"])
    ax.text(
        0.95, 0.1,
        f"Total: {total}\nSuccessful: {total - failed}\nFailed: {failed}",
        transform=ax.transAxes, va="bottom", ha="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    path = os.path.join(output_dir, "throughput.png")
    plt.savefig(path)
    plt.close()
    print(f"[plots] → {path}")


def plot_score_breakdown(report: dict, output_dir: str):
    """Plot per-request score breakdown (s_ttft, s_tpot, s_request)."""
    if not HAS_MPL:
        return

    per_req = report.get("per_request", [])
    if not per_req:
        return

    set_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    indices = list(range(len(per_req)))
    s_ttft = [r.get("s_ttft", 0) for r in per_req]
    s_tpot = [r.get("s_tpot", 0) for r in per_req]
    s_req = [r.get("s_request", 0) for r in per_req]

    width = 0.25
    ax.bar([i - width for i in indices], s_ttft, width, label="s_ttft", color="steelblue", alpha=0.8)
    ax.bar(indices, s_tpot, width, label="s_tpot", color="coral", alpha=0.8)
    ax.bar([i + width for i in indices], s_req, width, label="s_request", color="darkgreen", alpha=0.8)

    ax.set_xlabel("Request index")
    ax.set_ylabel("Score")
    ax.set_title("Per-Request Score Breakdown")
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    path = os.path.join(output_dir, "score_breakdown.png")
    plt.savefig(path)
    plt.close()
    print(f"[plots] → {path}")


def generate_all(report: dict, output_dir: str):
    """Generate all plots for a benchmark report."""
    if not HAS_MPL:
        print("[plots] matplotlib not installed; skipping plots")
        return
    plot_latency_timeline(report, output_dir)
    plot_latency_histogram(report, output_dir)
    plot_throughput_chart(report, output_dir)
    plot_score_breakdown(report, output_dir)


def main():
    p = argparse.ArgumentParser(description="Generate benchmark plots")
    p.add_argument("--result", required=True, help="benchmark_result.json")
    p.add_argument("--output-dir", default=".")
    args = p.parse_args()

    report = _load(args.result)
    generate_all(report, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
