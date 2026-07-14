"""
ERS (Effective Request Score) Benchmark Script
LLM Inference Optimization Challenge - Phase 1

Measures TTFT and TPOT for each request and computes ERS score.

Usage:
  python benchmark_ers.py --trace trace-round1.jsonl --endpoint http://localhost:8000/v1
  python benchmark_ers.py --trace trace-round1.jsonl --endpoint http://localhost:8000/v1 --model Qwen3.5-2B
"""
import argparse
import json
import time
import statistics
import urllib.request
from dataclasses import dataclass

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


@dataclass
class ERSConfig:
    """ERS scoring parameters from Phase 1 specification."""
    f_ttft: float = 0.1      # TTFT Floor = 100ms
    c_ttft: float = 1.5      # TTFT Ceiling = 1500ms
    f_tpot: float = 0.02     # TPOT Floor = 20ms
    c_tpot: float = 0.045    # TPOT Ceiling = 45ms
    gamma: float = 2.0       # Penalty curve steepness
    w: float = 0.5           # TTFT weight


@dataclass
class RequestResult:
    request_id: str
    success: bool
    ttft: float = 0.0
    tpot: float = 0.0
    num_output_tokens: int = 0
    error: str = ""


def make_request_httpx(endpoint: str, model: str, messages: list,
                       max_tokens: int, temperature: float = 0.0,
                       timeout_s: float = 300.0) -> tuple[float, float, int]:
    """Send request with httpx (faster streaming)."""
    import httpx as _httpx

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    t_start = time.time()
    t_first_token: float | None = None
    token_times: list[float] = []
    num_tokens = 0

    with _httpx.Client(timeout=_httpx.Timeout(timeout_s)) as client:
        with client.stream("POST", f"{endpoint}/chat/completions", json=payload) as resp:
            for line in resp.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line == "data: [DONE]":
                    break
                if not line.startswith("data: "):
                    continue

                json_str = line[6:]
                try:
                    data_obj = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

                choices = data_obj.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")

                now = time.time()
                if content:
                    if t_first_token is None:
                        t_first_token = now
                    token_times.append(now)
                    num_tokens += 1

    if t_first_token is None:
        return 0.0, 0.0, 0

    ttft = t_first_token - t_start

    if len(token_times) <= 1:
        tpot = ttft
    else:
        gaps = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
        tpot = statistics.mean(gaps)

    return ttft, tpot, num_tokens


def make_request_urllib(endpoint: str, model: str, messages: list,
                        max_tokens: int, temperature: float = 0.0,
                        timeout_s: float = 300.0) -> tuple[float, float, int]:
    """Fallback: send request with urllib."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_start = time.time()
    t_first_token: float | None = None
    token_times: list[float] = []
    num_tokens = 0

    resp = urllib.request.urlopen(req, timeout=timeout_s)
    # SSE parsing: accumulate bytes, split on newlines
    buf = b""
    while True:
        chunk = resp.read(1)
        if not chunk:
            break
        buf += chunk
        if b"\n" not in buf:
            continue

        line_bytes, buf = buf.split(b"\n", 1)
        line = line_bytes.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":") or line == "data: [DONE]":
            continue
        if not line.startswith("data: "):
            continue

        json_str = line[6:]
        try:
            data_obj = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        choices = data_obj.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        finish_reason = choices[0].get("finish_reason")

        now = time.time()
        if content:
            if t_first_token is None:
                t_first_token = now
            token_times.append(now)
            num_tokens += 1

    if t_first_token is None:
        return 0.0, 0.0, 0

    ttft = t_first_token - t_start
    if len(token_times) <= 1:
        tpot = ttft
    else:
        gaps = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
        tpot = statistics.mean(gaps)

    return ttft, tpot, num_tokens


def compute_s_score(value: float, floor: float, ceiling: float, gamma: float) -> float:
    """Compute normalized score [0,1] for TTFT or TPOT."""
    if value <= floor:
        return 1.0
    if value >= ceiling:
        return 0.0
    raw = (ceiling - value) / (ceiling - floor)
    return raw ** gamma


def compute_ers(results: list[RequestResult], config: ERSConfig) -> float:
    """Compute Effective Request Score over all results."""
    scores = []
    for r in results:
        if not r.success or r.num_output_tokens == 0:
            scores.append(0.0)
            continue
        s_ttft = compute_s_score(r.ttft, config.f_ttft, config.c_ttft, config.gamma)
        s_tpot = compute_s_score(r.tpot, config.f_tpot, config.c_tpot, config.gamma)
        s_request = config.w * s_ttft + (1 - config.w) * s_tpot
        scores.append(s_request)
    return statistics.mean(scores) if scores else 0.0


def load_trace(trace_path: str) -> list[dict]:
    """Load JSONL trace file."""
    requests = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                requests.append(json.loads(line))
    return requests


def main():
    parser = argparse.ArgumentParser(description="ERS Benchmark for LLM Inference")
    parser.add_argument("--trace", required=True, help="Path to trace JSONL file")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1",
                        help="OpenAI-compatible API endpoint")
    parser.add_argument("--model", default="Qwen3.5-2B", help="Model name")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Max concurrent requests (default=1 for sequential)")
    parser.add_argument("--output", default="ers_results.json", help="Output results file")
    args = parser.parse_args()

    config = ERSConfig()
    requests = load_trace(args.trace)
    print(f"📂 Loaded {len(requests)} requests from trace")

    make_request = make_request_httpx if HAS_HTTPX else make_request_urllib
    if HAS_HTTPX:
        print("✅ Using httpx for streaming requests")
    else:
        print("⚠️  httpx not available, using urllib fallback")

    results: list[RequestResult] = []
    failed = 0

    print(f"\n🚀 Starting ERS benchmark with {args.concurrency} concurrent workers...\n")

    for i, req in enumerate(requests):
        req_id = req.get("id", str(i))
        messages = req.get("messages", [])
        max_tokens = req.get("max_tokens", 512)

        print(f"[{i+1}/{len(requests)}] Request {req_id}: "
              f"messages={len(messages)}, max_tokens={max_tokens}")

        try:
            ttft, tpot, num_tokens = make_request(
                args.endpoint, args.model, messages, max_tokens
            )
            if num_tokens == 0:
                failed += 1
                results.append(RequestResult(req_id, False, error="Zero output tokens"))
                print(f"  ❌ Failed: Zero output tokens")
                continue

            results.append(RequestResult(req_id, True, ttft, tpot, num_tokens))
            s_ttft = compute_s_score(ttft, config.f_ttft, config.c_ttft, config.gamma)
            s_tpot = compute_s_score(tpot, config.f_tpot, config.c_tpot, config.gamma)
            s_req = config.w * s_ttft + (1 - config.w) * s_tpot
            print(f"  ✅ TTFT={ttft*1000:.1f}ms  TPOT={tpot*1000:.1f}ms  "
                  f"tokens={num_tokens}  score={s_req:.4f}")
        except Exception as e:
            failed += 1
            results.append(RequestResult(req_id, False, error=str(e)))
            print(f"  ❌ Error: {e}")

    # Aggregate
    successful = [r for r in results if r.success]
    ers = compute_ers(results, config)

    avg_ttft = statistics.mean([r.ttft for r in successful]) * 1000 if successful else 0.0
    avg_tpot = statistics.mean([r.tpot for r in successful]) * 1000 if successful else 0.0

    tpot_vals = sorted([r.tpot for r in successful])
    p50_tpot = statistics.median(tpot_vals) * 1000 if tpot_vals else 0.0
    n95 = int(len(tpot_vals) * 0.95)
    p95_tpot = (tpot_vals[n95] * 1000) if n95 < len(tpot_vals) else (tpot_vals[-1] * 1000 if tpot_vals else 0.0)

    # Per-token s_tpot stats
    s_tpot_vals = [
        compute_s_score(r.tpot, config.f_tpot, config.c_tpot, config.gamma)
        for r in successful
    ]
    avg_s_tpot = statistics.mean(s_tpot_vals) if s_tpot_vals else 0.0

    s_ttft_vals = [
        compute_s_score(r.ttft, config.f_ttft, config.c_ttft, config.gamma)
        for r in successful
    ]
    avg_s_ttft = statistics.mean(s_ttft_vals) if s_ttft_vals else 0.0

    # Save report
    report = {
        "ers": round(ers, 6),
        "config": {
            "f_ttft_ms": config.f_ttft * 1000,
            "c_ttft_ms": config.c_ttft * 1000,
            "f_tpot_ms": config.f_tpot * 1000,
            "c_tpot_ms": config.c_tpot * 1000,
            "gamma": config.gamma,
            "w": config.w,
        },
        "summary": {
            "total": len(results),
            "successful": len(successful),
            "failed": failed,
        },
        "metrics": {
            "avg_ttft_ms": round(avg_ttft, 2),
            "avg_tpot_ms": round(avg_tpot, 2),
            "median_tpot_ms": round(p50_tpot, 2),
            "p95_tpot_ms": round(p95_tpot, 2),
            "avg_s_ttft": round(avg_s_ttft, 4),
            "avg_s_tpot": round(avg_s_tpot, 4),
        },
        "score_breakdown": {
            "final_score_100xERS": round(100 * ers, 4),
            "ers_component_s_ttft_weighted": round(config.w * avg_s_ttft, 4),
            "ers_component_s_tpot_weighted": round((1 - config.w) * avg_s_tpot, 4),
        },
        "per_request": [
            {
                "id": r.request_id,
                "success": r.success,
                "ttft_ms": round(r.ttft * 1000, 2),
                "tpot_ms": round(r.tpot * 1000, 2),
                "num_tokens": r.num_output_tokens,
                "s_ttft": round(compute_s_score(r.ttft, config.f_ttft, config.c_ttft, config.gamma), 4),
                "s_tpot": round(compute_s_score(r.tpot, config.f_tpot, config.c_tpot, config.gamma), 4),
                "error": r.error if not r.success else None,
            }
            for r in results
        ],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Pretty print
    print(f"\n{'=' * 60}")
    print(f"  🏆 ERS BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"  📊 ERS:              {ers:.6f}")
    print(f"  📊 Score (100×ERS):  {100 * ers:.4f}")
    print(f"  {'─' * 40}")
    print(f"  ✅ Successful:       {len(successful)}/{len(results)}")
    print(f"  ❌ Failed:           {failed}")
    print(f"  {'─' * 40}")
    print(f"  ⏱  Avg TTFT:         {avg_ttft:.1f} ms")
    print(f"  ⚡ Avg TPOT:         {avg_tpot:.1f} ms")
    print(f"  ⚡ Median TPOT:      {p50_tpot:.1f} ms")
    print(f"  ⚡ P95 TPOT:         {p95_tpot:.1f} ms")
    print(f"  {'─' * 40}")
    print(f"  🎯 Avg s_TTFT:       {avg_s_ttft:.4f}")
    print(f"  🎯 Avg s_TPOT:       {avg_s_tpot:.4f}")
    print(f"  🎯 w*s_TTFT:         {config.w * avg_s_ttft:.4f}")
    print(f"  🎯 (1-w)*s_TPOT:     {(1 - config.w) * avg_s_tpot:.4f}")
    print(f"{'=' * 60}")
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
