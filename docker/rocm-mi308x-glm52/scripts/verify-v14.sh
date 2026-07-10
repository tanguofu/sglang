#!/bin/bash
# v14 verification script — MTP + HiCache + smoke + long context
# Usage: verify-v14.sh [NODE_IP]
set -euo pipefail

NODE_IP="${1:-127.0.0.1}"
PORT=30000
API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
BASE="http://${NODE_IP}:${PORT}"

echo "============================================"
echo "  v14 Verification — ${NODE_IP}:${PORT}"
echo "  MTP + HiCache + FP8 KV + DSA fused-store disabled"
echo "============================================"
echo ""

# ---- 1. Health check ----
echo "[1/5] Health check..."
HEALTH=$(curl -s --connect-timeout 5 --max-time 10 "${BASE}/health" 2>/dev/null || echo "FAIL")
if [ "$HEALTH" = "ok" ] || [ "$HEALTH" = "true" ]; then
    echo "  ✓ Health: ${HEALTH}"
else
    echo "  ✗ Health: ${HEALTH}"
    exit 1
fi
echo ""

# ---- 2. Metrics: MTP accept rate + HiCache max_total ----
echo "[2/5] Metrics (MTP + HiCache)..."
METRICS=$(curl -s --connect-timeout 5 --max-time 10 "${BASE}/metrics" 2>/dev/null || echo "")

# MTP accept rate
ACCEPT_LEN=$(echo "$METRICS" | grep -E "^sglang_spec_accept_length" | grep -v "^#" | head -1 | awk '{print $NF}')
DRAFT_LEN=$(echo "$METRICS" | grep -E "^sglang_spec_drafted_length" | grep -v "^#" | head -1 | awk '{print $NF}')
echo "  MTP accept_length: ${ACCEPT_LEN:-N/A}"
echo "  MTP drafted_length: ${DRAFT_LEN:-N/A}"

# HiCache / KV cache capacity
MAX_TOTAL=$(echo "$METRICS" | grep -E "^sglang_max_total_num_tokens" | grep -v "^#" | head -1 | awk '{print $NF}')
CACHE_HIT=$(echo "$METRICS" | grep -E "^sglang_hicache_hit" | grep -v "^#" | head -1 | awk '{print $NF}')
echo "  max_total_num_tokens: ${MAX_TOTAL:-N/A}"
echo "  hicache_hit: ${CACHE_HIT:-N/A}"

# Check if max_total > 1M (1048576)
if [ -n "$MAX_TOTAL" ]; then
    if [ "$MAX_TOTAL" -gt 1048576 ]; then
        echo "  ✓ HiCache: max_total (${MAX_TOTAL}) > 1M (1048576) — 1M context supported"
    else
        echo "  ⚠ HiCache: max_total (${MAX_TOTAL}) <= 1M — may not support full 1M context"
    fi
fi
echo ""

# ---- 3. Short prompt smoke test ----
echo "[3/5] Short prompt smoke test..."
SMOKE=$(curl -s --connect-timeout 10 --max-time 60 "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is 2+3? Answer with just the number."}],"max_tokens":16,"temperature":0}' 2>/dev/null || echo "FAIL")

SMOKE_CONTENT=$(echo "$SMOKE" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])' 2>/dev/null || echo "PARSE_FAIL")
if [ "$SMOKE_CONTENT" != "PARSE_FAIL" ] && [ -n "$SMOKE_CONTENT" ]; then
    echo "  ✓ Smoke test response: ${SMOKE_CONTENT}"
else
    echo "  ✗ Smoke test failed: ${SMOKE:0:200}"
fi
echo ""

# ---- 4. Long context test (>2K tokens, verify no garbled output) ----
echo "[4/5] Long context test (>2K tokens)..."
# Generate a ~3K token prompt by repeating a paragraph
LONG_TEXT=$(python3 -c 'print("The quick brown fox jumps over the lazy dog. " * 200)')
LONG_PAYLOAD=$(python3 -c "
import json
text = 'The quick brown fox jumps over the lazy dog. ' * 200
print(json.dumps({
    'model': 'glm-5.2',
    'messages': [{'role': 'user', 'content': text + 'What animal is mentioned in the text above? Answer in one word.'}],
    'max_tokens': 32,
    'temperature': 0
}))
")

LONG_START=$(date +%s)
LONG_RESP=$(curl -s --connect-timeout 10 --max-time 120 "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d "$LONG_PAYLOAD" 2>/dev/null || echo "FAIL")
LONG_END=$(date +%s)
LONG_DURATION=$((LONG_END - LONG_START))

LONG_CONTENT=$(echo "$LONG_RESP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])' 2>/dev/null || echo "PARSE_FAIL")
LONG_PROMPT_TOKENS=$(echo "$LONG_RESP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("usage",{}).get("prompt_tokens","?"))' 2>/dev/null || echo "?")
if [ "$LONG_CONTENT" != "PARSE_FAIL" ] && [ -n "$LONG_CONTENT" ]; then
    echo "  ✓ Long context response: ${LONG_CONTENT}"
    echo "  ✓ Prompt tokens: ${LONG_PROMPT_TOKENS}, duration: ${LONG_DURATION}s"
else
    echo "  ✗ Long context failed: ${LONG_RESP:0:200}"
fi
echo ""

# ---- 5. MTP accept rate (after requests) ----
echo "[5/5] MTP accept rate (post-request)..."
sleep 2
METRICS2=$(curl -s --connect-timeout 5 --max-time 10 "${BASE}/metrics" 2>/dev/null || echo "")
ACCEPT_LEN2=$(echo "$METRICS2" | grep -E "^sglang_spec_accept_length" | grep -v "^#" | head -1 | awk '{print $NF}')
DRAFT_LEN2=$(echo "$METRICS2" | grep -E "^sglang_spec_drafted_length" | grep -v "^#" | head -1 | awk '{print $NF}')
echo "  MTP accept_length: ${ACCEPT_LEN2:-N/A}"
echo "  MTP drafted_length: ${DRAFT_LEN2:-N/A}"

if [ -n "$ACCEPT_LEN2" ] && [ -n "$DRAFT_LEN2" ] && [ "$DRAFT_LEN2" != "0" ]; then
    ACCEPT_RATE=$(python3 -c "print(round(${ACCEPT_LEN2}/${DRAFT_LEN2}, 3))")
    echo "  MTP accept_rate: ${ACCEPT_RATE}"
    if [ "$(python3 -c "print(1 if ${ACCEPT_RATE} > 0.5 else 0)")" = "1" ]; then
        echo "  ✓ MTP accept_rate > 0.5 — healthy"
    else
        echo "  ⚠ MTP accept_rate <= 0.5 — may need tuning"
    fi
else
    echo "  ⚠ MTP metrics not yet populated (need more requests)"
fi
echo ""

echo "============================================"
echo "  Verification complete — ${NODE_IP}"
echo "============================================"
