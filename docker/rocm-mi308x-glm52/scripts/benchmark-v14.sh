#!/bin/bash
# Benchmark script for v14 — MTP enabled, compare to 355X
# Usage: benchmark-v14.sh [NODE_IP] [LABEL]
set -euo pipefail

NODE_IP="${1:-127.0.0.1}"
LABEL="${2:-test-0}"
PORT=30000
API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
BASE="http://${NODE_IP}:${PORT}"
OUTDIR="/tmp/bench-${LABEL}"
mkdir -p "$OUTDIR"

echo "============================================"
echo "  Benchmark v14 — ${LABEL} (${NODE_IP})"
echo "  MTP: steps=3, draft_tokens=4, eagle_topk=1"
echo "============================================"

# ---- Benchmark 1: short_c32 (32 input, 256 output) ----
echo ""
echo "[1/4] short_c32 (input=32, output=256, concurrency=8)..."
python3 -m sglang.bench_serving \
    --base-url "${BASE}" \
    --api-key "${API_KEY}" \
    --model "glm-5.2" \
    --num-prompts 32 \
    --request-rate 8 \
    --input 32 \
    --output 256 \
    --output-file "${OUTDIR}/short_c32.json" 2>&1 | tee "${OUTDIR}/short_c32.log" | tail -20

# ---- Benchmark 2: short_c128 (128 input, 256 output) ----
echo ""
echo "[2/4] short_c128 (input=128, output=256, concurrency=8)..."
python3 -m sglang.bench_serving \
    --base-url "${BASE}" \
    --api-key "${API_KEY}" \
    --model "glm-5.2" \
    --num-prompts 32 \
    --request-rate 8 \
    --input 128 \
    --output 256 \
    --output-file "${OUTDIR}/short_c128.json" 2>&1 | tee "${OUTDIR}/short_c128.log" | tail -20

# ---- Benchmark 3: mid_c32 (2048 input, 256 output) ----
echo ""
echo "[3/4] mid_c2048 (input=2048, output=256, concurrency=4)..."
python3 -m sglang.bench_serving \
    --base-url "${BASE}" \
    --api-key "${API_KEY}" \
    --model "glm-5.2" \
    --num-prompts 16 \
    --request-rate 4 \
    --input 2048 \
    --output 256 \
    --output-file "${OUTDIR}/mid_c2048.json" 2>&1 | tee "${OUTDIR}/mid_c2048.log" | tail -20

# ---- Benchmark 4: MTP metrics snapshot ----
echo ""
echo "[4/4] MTP metrics snapshot..."
curl -s "${BASE}/metrics" 2>/dev/null | grep -E "sglang_spec_|sglang_max_total|sglang_hicache" | grep -v "^#" > "${OUTDIR}/mtp_metrics.txt"
cat "${OUTDIR}/mtp_metrics.txt"

echo ""
echo "============================================"
echo "  Benchmark complete — ${LABEL}"
echo "  Results saved to ${OUTDIR}/"
echo "============================================"
