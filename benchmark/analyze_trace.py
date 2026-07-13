#!/usr/bin/env python3
"""Phase 2: Deep trace analysis — find optimisation opportunities."""

import argparse
import json
import os
import sys
from collections import Counter

from .parser import load_trace, trace_statistics


def _fmt(v: float, unit: str = "") -> str:
    if unit:
        return f"{v:.2f} {unit}"
    return f"{v:.2f}"


def repeat_prefix_analysis(requests: list) -> dict:
    """
    Detect repeated message prefixes across requests.
    This helps decide if prefix caching would help.
    """
    # Get the first user message content from each request
    first_messages = []
    for req in requests:
        if req.messages and len(req.messages) > 0:
            first_messages.append(req.messages[0].get("content", ""))
        else:
            first_messages.append("")

    # Count exact duplicate first messages
    msg_counts = Counter(first_messages)

    # Count shared first N-char prefixes
    prefix_sharing = {}
    for length in [10, 50, 100, 500, 1000]:
        pref_counts = Counter(m[:length] for m in first_messages if len(m) >= length)
        if pref_counts:
            shared = sum(c - 1 for c in pref_counts.values() if c > 1)
            prefix_sharing[str(length)] = {
                "unique_prefixes": len(pref_counts),
                "shared_beyond_first": shared,
            }

    # Cache hit ratio estimate (if prefix cache were used)
    total_savable = 0
    longest_common_prefix = 0
    if first_messages:
        # Sort by length and find LCP of consecutive sorted messages
        sorted_msgs = sorted(first_messages)
        for i in range(1, len(sorted_msgs)):
            common = 0
            a, b = sorted_msgs[i - 1], sorted_msgs[i]
            for ca, cb in zip(a, b):
                if ca == cb:
                    common += 1
                else:
                    break
            longest_common_prefix = max(longest_common_prefix, common)

    return {
        "total_requests": len(requests),
        "unique_first_messages": len(msg_counts),
        "most_common_first_message_count": msg_counts.most_common(1)[0][1] if msg_counts else 0,
        "longest_shared_prefix_chars": longest_common_prefix,
        "prefix_sharing_by_length": prefix_sharing,
    }


def burst_analysis(requests: list) -> dict:
    """Analyse burst/idle patterns in the trace."""
    arrivals = [r.arrival_time for r in requests]
    gaps = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))]

    if not gaps:
        return {"bursts": 0, "avg_gap": 0}

    mean_gap = sum(gaps) / len(gaps)

    # Find burst windows: consecutive arrivals with gaps < mean/2
    bursts = []
    current_burst: list[float] = [arrivals[0]]
    for i in range(1, len(arrivals)):
        gap = arrivals[i] - arrivals[i - 1]
        if gap < mean_gap / 2:
            current_burst.append(arrivals[i])
        else:
            if len(current_burst) > 2:
                bursts.append(current_burst)
            current_burst = [arrivals[i]]
    if len(current_burst) > 2:
        bursts.append(current_burst)

    # Idle periods: gaps significantly above mean
    idle_periods = [g for g in gaps if g > mean_gap * 2]

    return {
        "num_bursts": len(bursts),
        "burst_sizes": [len(b) for b in bursts],
        "max_burst_size": max((len(b) for b in bursts), default=0),
        "idle_periods": len(idle_periods),
        "idle_total_time_s": sum(idle_periods) if idle_periods else 0,
        "idle_pct": sum(idle_periods) / (arrivals[-1] - arrivals[0]) * 100 if (idle_periods and len(arrivals) > 1) else 0,
        "mean_gap_s": mean_gap,
    }


def generate_analysis(requests: list, output_dir: str = "."):
    """Generate full trace analysis with optimisation recommendations."""
    stats = trace_statistics(requests)
    prefix = repeat_prefix_analysis(requests)
    bursts = burst_analysis(requests)

    # Build markdown report
    md = []
    md.append("# Trace Analysis Report\n")
    md.append("## Overview\n")
    md.append(f"| Metric | Value |")
    md.append(f"|---|---|")
    md.append(f"| Total requests | {stats['num_requests']} |")
    md.append(f"| Trace duration | {stats['total_duration_s']:.1f}s |")
    md.append(f"| Arrival rate | {stats['arrival_rate_req_per_s']:.2f} req/s |")
    md.append(f"| Max concurrent (60s window) | {stats['max_concurrent_60s_window']} |")
    md.append("")

    md.append("## Arrival Pattern\n")
    gap_s = stats.get("inter_arrival_gap_s", {})
    md.append(f"| Metric | Mean | Min | Max | Std |")
    md.append(f"|---|---|---|---|---|")
    md.append(f"| Inter-arrival gap (s) | {gap_s.get('mean', 0):.2f} | {gap_s.get('min', 0):.2f} | {gap_s.get('max', 0):.2f} | {gap_s.get('stdev', 0):.2f} |")
    md.append("")

    md.append("## Burst / Idle Analysis\n")
    md.append(f"- **Burst count:** {bursts['num_bursts']}")
    md.append(f"- **Max burst size:** {bursts['max_burst_size']} requests")
    md.append(f"- **Burst sizes:** {bursts['burst_sizes']}")
    md.append(f"- **Idle periods:** {bursts['idle_periods']} (total {bursts['idle_total_time_s']:.1f}s, {bursts['idle_pct']:.1f}% of trace)")
    md.append(f"- **Mean gap:** {bursts['mean_gap_s']:.2f}s")
    md.append("")

    md.append("## Prompt & Output Length\n")
    prompt_s = stats.get("prompt_length_chars", {})
    output_s = stats.get("expected_output_tokens", {})
    md.append("| Metric | Mean | Min | Max | Std |")
    md.append("|---|---|---|---|---|")
    md.append(f"| Prompt length (chars) | {prompt_s.get('mean',0):.0f} | {prompt_s.get('min',0):.0f} | {prompt_s.get('max',0):.0f} | {prompt_s.get('stdev',0):.0f} |")
    md.append(f"| Expected output (tokens) | {output_s.get('mean',0):.1f} | {output_s.get('min',0):.0f} | {output_s.get('max',0):.0f} | {output_s.get('stdev',0):.0f} |")
    md.append("")

    md.append("## Prefix Caching Analysis\n")
    md.append(f"- **Unique first messages:** {prefix['unique_first_messages']} / {prefix['total_requests']} requests")
    md.append(f"- **Most repeated first message:** {prefix['most_common_first_message_count']}×")
    md.append(f"- **Longest shared prefix:** {prefix['longest_shared_prefix_chars']} chars")
    md.append(f"- **Prefix sharing by length:**")
    for length, data in sorted(prefix['prefix_sharing_by_length'].items(), key=lambda x: int(x[0])):
        md.append(f"  - {length}-char prefix: {data['unique_prefixes']} unique variants, {data['shared_beyond_first']} cache-saveable requests")
    md.append("")

    md.append("## Optimisation Recommendations\n")
    recommendations = []

    # 1. Prefix caching
    total_savable = sum(
        d.get("shared_beyond_first", 0)
        for d in prefix["prefix_sharing_by_length"].values()
    )
    if total_savable > 0:
        recommendations.append(
            "- **Enable prefix caching** — shared prefixes detected across requests."
        )
    if prefix["longest_shared_prefix_chars"] > 100:
        recommendations.append(
            "- **Prefix caching expected to be effective** — long shared prefixes (up to "
            f"{prefix['longest_shared_prefix_chars']} chars)."
        )

    # 2. Batch sizing
    max_concurrent = stats['max_concurrent_60s_window']
    if max_concurrent > 10:
        recommendations.append(
            f"- **Set --max-num-seqs >= {max_concurrent}** — peak concurrency is {max_concurrent}."
        )
    recommendations.append(
        f"- **Set --gpu-memory-utilization** to 0.95 to max VRAM usage."
    )

    # 3. TPOT sensitivity
    output_mean = output_s.get("mean", 0)
    output_max = output_s.get("max", 0)
    recommendations.append(
        f"- **TPOT is the critical metric** — floor=20ms, ceiling=45ms. "
        f"Output tokens up to {output_max:.0f} per request."
    )
    if output_max > 256:
        recommendations.append(
            "- **Consider speculative decoding** for long generation requests."
        )

    # 4. Burst handling
    if bursts["num_bursts"] > 0:
        recommendations.append(
            f"- **Burst pattern detected** ({bursts['num_bursts']} bursts). "
            "Consider enabling chunked prefill and dynamic batching."
        )

    # 5. Idle periods
    if bursts["idle_pct"] > 30:
        recommendations.append(
            f"- **{bursts['idle_pct']:.0f}% idle time** — consider right-sizing or "
            "sharing the instance."
        )

    recommendations.append(
        "- **Enable FP8 quantization** for KV cache and weights to free VRAM."
    )

    md.append("\n".join(recommendations))
    md.append("")

    # Write
    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, "trace_analysis.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"[analyze_trace] report → {md_path}")

    # Also save structured data
    structured = {
        "statistics": stats,
        "prefix_analysis": prefix,
        "burst_analysis": bursts,
    }
    struct_path = os.path.join(output_dir, "trace_data.json")
    with open(struct_path, "w") as f:
        json.dump(structured, f, indent=2)
    print(f"[analyze_trace] structured data → {struct_path}")

    return structured


def main():
    p = argparse.ArgumentParser(description="Analyse trace-round1.jsonl")
    p.add_argument("--trace", required=True, help="Path to trace-round1.jsonl")
    p.add_argument("--output-dir", default=".")
    args = p.parse_args()

    requests = load_trace(args.trace)
    print(f"[analyze_trace] loaded {len(requests)} requests")

    generate_analysis(requests, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
