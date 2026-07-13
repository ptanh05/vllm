"""Async HTTP client for OpenAI-compatible LLM serving endpoint."""

import json
import time
import statistics
import httpx
from .metrics import RequestTiming


async def stream_completion(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout_s: float = 600.0,
) -> RequestTiming:
    """Stream a chat completion, measuring TTFT and per-token TPOT.

    Returns a RequestTiming with:
      - ttft_s: time to first token
      - tpot_list_s: list of per-token latencies
      - num_output_tokens
      - total_latency_s
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    token_times: list[float] = []
    t_first_token: float | None = None
    t_start = time.monotonic()

    try:
        async with client.stream(
            "POST",
            f"{endpoint}/chat/completions",
            json=payload,
            timeout=httpx.Timeout(timeout_s),
        ) as resp:
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":") or line == "data: [DONE]":
                    continue
                if not line.startswith("data: "):
                    continue

                json_str = line[6:]
                try:
                    data = json.loads(json_str)
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    finish_reason = choices[0].get("finish_reason")

                    now = time.monotonic()
                    if content:
                        if t_first_token is None:
                            t_first_token = now
                        token_times.append(now)

                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return RequestTiming(error=str(e))

    t_end = time.monotonic()

    if t_first_token is None:
        return RequestTiming(error="no tokens received", total_latency_s=t_end - t_start)

    ttft = t_first_token - t_start
    total_latency = t_end - t_start

    return RequestTiming(
        ttft_s=ttft,
        tpot_list_s=token_times,
        num_output_tokens=len(token_times),
        total_latency_s=total_latency,
        success=True,
    )


def build_messages_text(prompt: str) -> list[dict]:
    """Wrap a flat prompt string into the messages format."""
    return [{"role": "user", "content": prompt}]
