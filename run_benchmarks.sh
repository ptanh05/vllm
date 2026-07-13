#!/bin/bash
# LLM Inference Optimization Challenge — Full Benchmark Suite
# Uses new benchmark/ package
set -euo pipefail

ENDPOINT="${ENDPOINT:-http://localhost:8000/v1}"
MODEL="${MODEL:-Qwen3.5-2B}"
TRACE="${TRACE:-trace-round1.jsonl}"
OUTPUT="${OUTPUT:-benchmark_output}"
GPQA_SCRIPT="${GPQA_SCRIPT:-benchmark_gpqa.py}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUTPUT"

echo "========================================="
echo "  LLM Inference Optimization Benchmark"
echo "  Model: $MODEL"
echo "  Endpoint: $ENDPOINT"
echo "  Trace: $TRACE"
echo "========================================="
echo ""

# Step 0: Health check
echo "[0/5] Checking server health..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")
if [ "$HTTP_CODE" != "200" ]; then
    echo "  Server not ready. Start with: docker compose up -d"
    echo "  Then wait ~60s for model to load."
    exit 1
fi
echo "  ✓ Server healthy"

# Step 1: Trace analysis
echo ""
echo "[1/5] Analyzing trace..."
if [ -f "$TRACE" ]; then
    $PYTHON -m benchmark.analyze_trace --trace "$TRACE" --output-dir "$OUTPUT"
    echo "  ✓ Trace analysis saved to $OUTPUT/trace_analysis.md"
else
    echo "  ⚠ Trace file $TRACE not found — skipping analysis"
fi

# Step 2: Warmup
echo ""
echo "[2/5] Warming up model..."
curl -s -X POST "$ENDPOINT/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 10}" \
    > /dev/null 2>&1 && echo "  ✓ Warmup complete" || echo "  ⚠ Warmup failed"

# Step 3: ERS benchmark
echo ""
echo "[3/5] Running ERS benchmark..."
if [ -f "$TRACE" ]; then
    $PYTHON -m benchmark.benchmark \
        --trace "$TRACE" \
        --endpoint "$ENDPOINT" \
        --model "$MODEL" \
        --concurrency 8 \
        --output-dir "$OUTPUT"
    echo "  ✓ ERS benchmark complete"
else
    echo "  ⚠ No trace file — running simple test"
    curl -s -X POST "$ENDPOINT/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? Answer briefly.\"}], \"max_tokens\": 50}" \
        | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print('Response:', d['choices'][0]['message']['content'][:80])"
fi

# Step 4: Generate plots
echo ""
echo "[4/5] Generating plots..."
if [ -f "$OUTPUT/benchmark_result.json" ]; then
    $PYTHON -m benchmark.plots --result "$OUTPUT/benchmark_result.json" --output-dir "$OUTPUT" 2>/dev/null || \
    echo "  ⚠ Plots skipped (install matplotlib for plots)"
fi

# Step 5: GPQA accuracy (optional)
echo ""
echo "[5/5] GPQA accuracy test..."
if [ -f "$GPQA_SCRIPT" ]; then
    $PYTHON "$GPQA_SCRIPT" --endpoint "$ENDPOINT" --model "$MODEL" --output "$OUTPUT/gpqa_results.json" 2>&1 || \
    echo "  ⚠ GPQA test failed"
else
    echo "  ⚠ $GPQA_SCRIPT not found — skipping accuracy test"
fi

echo ""
echo "========================================="
echo "  BENCHMARK COMPLETE"
echo "========================================="
echo "  Results in: $OUTPUT/"
ls -lh "$OUTPUT/" 2>/dev/null || true
echo ""
echo "  Key files:"
echo "    - $OUTPUT/trace_analysis.md"
echo "    - $OUTPUT/benchmark_result.json"
echo "    - $OUTPUT/benchmark_analysis.md"
echo "    - $OUTPUT/gpqa_results.json"
echo "    - $OUTPUT/*.png (plots)"
echo "========================================="
