"""Trace parser — reads trace-round1.jsonl and normalises fields."""

import json
import statistics as _stats
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceRequest:
    """A single request from the trace file."""
    request_id: str = ""
    arrival_time: float = 0.0  # seconds from trace start
    messages: list[dict] = field(default_factory=list)
    max_tokens: int = 256
    temperature: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_length(self) -> int:
        total = 0
        for m in self.messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total += len(content)
        return total

    @property
    def expected_output_length(self) -> int:
        return self.max_tokens


def load_trace(path_or_lines: str | list[str]) -> list[TraceRequest]:
    """Load a trace from a JSONL file path or a list of JSON lines."""
    if isinstance(path_or_lines, str):
        with open(path_or_lines, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
    else:
        lines = path_or_lines

    requests: list[TraceRequest] = []
    for i, line in enumerate(lines):
        obj = json.loads(line)
        req = TraceRequest(
            request_id=obj.get("id", obj.get("request_id", f"req-{i:04d}")),
            arrival_time=float(obj.get("arrival_time", obj.get("timestamp", 0.0))),
            messages=obj.get("messages", obj.get("prompt", [])),
            max_tokens=int(obj.get("max_tokens", 256)),
            temperature=float(obj.get("temperature", 0.0)),
            raw=obj,
        )
        requests.append(req)

    requests.sort(key=lambda r: r.arrival_time)
    return requests


def _stats_dict(vals: list[float]) -> dict:
    if not vals:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "count": 0}
    d = {
        "min": min(vals),
        "max": max(vals),
        "mean": _stats.mean(vals),
        "median": _stats.median(vals),
        "count": len(vals),
    }
    if len(vals) > 1:
        d["stdev"] = _stats.stdev(vals)
    else:
        d["stdev"] = 0
    return d


def trace_statistics(requests: list[TraceRequest]) -> dict:
    """Compute aggregate statistics over the trace."""
    if not requests:
        return {}

    arrivals = [r.arrival_time for r in requests]
    prompt_lens = [r.prompt_length for r in requests]
    output_lens = [r.expected_output_length for r in requests]

    gaps = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))]
    total_duration = arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 1.0

    burst_threshold = _stats.mean(gaps) if gaps else 0
    burst_intervals = [g for g in gaps if g < burst_threshold]
    idle_intervals = [g for g in gaps if g >= burst_threshold]

    # Estimate concurrent requests within a 60-second sliding window
    concurrent = []
    for t in arrivals:
        count = sum(1 for a in arrivals if a <= t < a + 60)
        concurrent.append(count)

    return {
        "num_requests": len(requests),
        "total_duration_s": total_duration,
        "arrival_rate_req_per_s": len(requests) / total_duration if total_duration > 0 else 0,
        "arrival_time_s": _stats_dict(arrivals),
        "inter_arrival_gap_s": _stats_dict(gaps),
        "burst_intervals_s": _stats_dict(burst_intervals),
        "idle_intervals_s": _stats_dict(idle_intervals),
        "prompt_length_chars": _stats_dict(prompt_lens),
        "expected_output_tokens": _stats_dict(output_lens),
        "concurrent_requests_60s_window": _stats_dict(concurrent),
        "max_concurrent_60s_window": max(concurrent) if concurrent else 0,
    }
