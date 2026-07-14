"""
GPQA Diamond Accuracy Benchmark
LLM Inference Optimization Challenge - Phase 1

Measures accuracy drop Δ = baseline_accuracy - team_accuracy and computes
the accuracy penalty f(Δ) applied to the final score.

Usage:
  python benchmark_gpqa.py --questions gpqa_diamond_100.jsonl --endpoint http://localhost:8000/v1
  python benchmark_gpqa.py --questions gpqa_diamond_100.jsonl --endpoint http://localhost:8000/v1 --model Qwen3.5-2B --baseline 0.4
"""
import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
# No extra imports needed - Python 3.12 uses X | None syntax

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


@dataclass
class GPQAExample:
    """A single GPQA Diamond question."""
    id: str
    question: str
    choices: dict[str, str]     # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_answer: str         # "A", "B", "C", or "D"
    explanation: str = ""


@dataclass
class GPQAResult:
    id: str
    model_answer: str
    correct_answer: str
    is_correct: bool
    response_text: str = ""
    error: str = ""


def parse_question(q: dict) -> GPQAExample:
    """Parse a GPQA question from JSON dict."""
    choices_raw = q.get("choices", q.get("options", {}))
    # Normalize choices format
    if isinstance(choices_raw, dict):
        choices = choices_raw
    elif isinstance(choices_raw, list):
        letters = ["A", "B", "C", "D"]
        choices = {}
        for i, c in enumerate(choices_raw):
            if i < len(letters):
                # Handle {"label": "A", "text": "..."} format
                if isinstance(c, dict):
                    choices[c.get("label", letters[i])] = c.get("text", c.get("content", ""))
                else:
                    choices[letters[i]] = c
            else:
                choices[letters[i]] = c
    else:
        choices = {}

    return GPQAExample(
        id=q.get("id", q.get("question_id", str(hash(q.get("question", ""))))),
        question=q.get("question", ""),
        choices=choices,
        correct_answer=q.get("answer", q.get("correct_answer", "")),
        explanation=q.get("explanation", ""),
    )


def build_prompt(ex: GPQAExample) -> str:
    """Build a multiple-choice prompt from a GPQA question."""
    choices_text = "\n".join(f"{k}) {v}" for k, v in ex.choices.items())
    prompt = (
        f"Question: {ex.question}\n\n"
        f"Choices:\n{choices_text}\n\n"
        f"Please reason step by step, then output your final answer as a single letter "
        f"(A, B, C, or D) inside \\boxed{{}}. Example: \\boxed{{A}}"
    )
    return prompt


def extract_answer(response_text: str) -> str | None:
    """Extract the answer letter from model response using multiple strategies."""
    # Strategy 1: \boxed{X}
    m = re.search(r'\\boxed\{([A-Da-d])\}', response_text)
    if m:
        return m.group(1).upper()

    # Strategy 2: "Answer: X"
    m = re.search(r'(?:Answer|answer|\\boxed)\s*:\s*\(?([A-Da-d])\)?', response_text)
    if m:
        return m.group(1).upper()

    # Strategy 3: "The correct answer is X"
    m = re.search(r'(?:correct answer|correct option|final answer)\s+is\s+[\(]?([A-Da-d])[\)]?',
                  response_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Strategy 4: Single letter near end, in brackets like (A) or [B]
    m = re.findall(r'[\(\[][A-Da-d][\)\]]', response_text[-200:])
    if m:
        return m[-1][-2].upper()

    return None


def query_model(endpoint: str, model: str, messages: list,
                max_tokens: int = 1024, temperature: float = 0.0,
                timeout_s: float = 120.0) -> str:
    """Query model and return the response text."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if HAS_HTTPX:
        with httpx.Client(timeout=httpx.Timeout(timeout_s)) as client:
            resp = client.post(f"{endpoint}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    else:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{endpoint}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]


def compute_accuracy_penalty(delta: float) -> float:
    """
    Compute accuracy penalty f(Δ).

    f(Δ) = 1.0                 if Δ <= 0.10
    f(Δ) = 1.0 - (Δ - 0.10)/0.06  if 0.10 < Δ < 0.16
    f(Δ) = 0.0                 if Δ >= 0.16
    """
    if delta <= 0.10:
        return 1.0
    elif delta >= 0.16:
        return 0.0
    else:
        return 1.0 - (delta - 0.10) / 0.06


def main():
    parser = argparse.ArgumentParser(description="GPQA Diamond Accuracy Benchmark")
    parser.add_argument("--questions", required=True, help="Path to GPQA Diamond JSONL file")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1",
                        help="OpenAI-compatible API endpoint")
    parser.add_argument("--model", default="Qwen3.5-2B", help="Model name")
    parser.add_argument("--baseline", type=float, default=0.4,
                        help="Reference BF16 accuracy (default: 0.4)")
    parser.add_argument("--output", default="gpqa_results.json", help="Output file")
    parser.add_argument("--max-questions", type=int, default=100,
                        help="Max questions to evaluate (default: 100)")
    args = parser.parse_args()

    # Load questions
    questions = []
    with open(args.questions, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(parse_question(json.loads(line)))

    if args.max_questions and len(questions) > args.max_questions:
        questions = questions[:args.max_questions]

    print(f"📂 Loaded {len(questions)} GPQA Diamond questions")
    print(f"🎯 Baseline accuracy: {args.baseline}")
    print(f"🚀 Model: {args.model} at {args.endpoint}\n")

    results: list[GPQAResult] = []
    correct = 0

    for i, q in enumerate(questions):
        prompt = build_prompt(q)
        messages = [
            {"role": "user", "content": prompt},
        ]

        print(f"[{i+1}/{len(questions)}] {q.id}...", end=" ", flush=True)

        try:
            response_text = query_model(args.endpoint, args.model, messages)
            model_answer = extract_answer(response_text)

            if model_answer is None:
                print(f"⚠️  No answer extracted (response: {response_text[:80]}...)")
                results.append(GPQAResult(
                    id=q.id, model_answer="", correct_answer=q.correct_answer,
                    is_correct=False, response_text=response_text[:200],
                    error="Could not extract answer"
                ))
                continue

            is_correct = model_answer == q.correct_answer.upper()
            if is_correct:
                correct += 1
                print(f"✅ {model_answer}")
            else:
                print(f"❌ Model: {model_answer}, Expected: {q.correct_answer}")

            results.append(GPQAResult(
                id=q.id, model_answer=model_answer, correct_answer=q.correct_answer,
                is_correct=is_correct, response_text=response_text[:200]
            ))

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append(GPQAResult(
                id=q.id, model_answer="", correct_answer=q.correct_answer,
                is_correct=False, error=str(e)
            ))

    # Compute metrics
    total_evaluated = len(results)
    accuracy = correct / total_evaluated if total_evaluated > 0 else 0.0
    delta = args.baseline - accuracy
    penalty = compute_accuracy_penalty(delta)

    report = {
        "baseline_accuracy": args.baseline,
        "team_accuracy": round(accuracy, 4),
        "delta": round(delta, 4),
        "penalty_f_delta": round(penalty, 4),
        "passed_gate": penalty > 0.0,
        "summary": {
            "total": total_evaluated,
            "correct": correct,
            "incorrect": total_evaluated - correct,
        },
        "accuracy_gate": {
            "threshold_delta_pass": "Δ <= 0.10 → f(Δ) = 1.0",
            "threshold_delta_partial": "0.10 < Δ < 0.16 → linear penalty",
            "threshold_delta_fail": "Δ >= 0.16 → f(Δ) = 0.0",
        },
        "per_question": [
            {
                "id": r.id,
                "model_answer": r.model_answer,
                "correct_answer": r.correct_answer,
                "is_correct": r.is_correct,
                "error": r.error if r.error else None,
            }
            for r in results
        ],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Pretty print
    print(f"\n{'=' * 60}")
    print(f"  🏆 GPQA DIAMOND ACCURACY RESULTS")
    print(f"{'=' * 60}")
    print(f"  📊 Baseline accuracy:  {args.baseline:.4f}")
    print(f"  📊 Team accuracy:      {accuracy:.4f}  ({correct}/{total_evaluated})")
    print(f"  📊 Δ (drop):           {delta:.4f}")
    print(f"  {'─' * 40}")
    print(f"  🎯 Penalty f(Δ):       {penalty:.4f}")
    print(f"  {'✅ PASSED' if penalty > 0.0 else '❌ FAILED'} accuracy gate (f(Δ) > 0)")
    print(f"  {'─' * 40}")
    print(f"  • Δ ≤ 0.10 → f(Δ)=1.0 (no penalty)")
    print(f"  • 0.10 < Δ < 0.16 → f(Δ) = {penalty:.4f}")
    print(f"  • Δ ≥ 0.16 → f(Δ)=0.0 (eliminated)")
    print(f"{'=' * 60}")
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
