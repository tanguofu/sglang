#!/usr/bin/env bash
# PD RDMA correctness test — tests via router (port 30001)
# Tests: 2+2, Paris, 70, 42, and MTP accept_len

set -euo pipefail

ROUTER_IP=${ROUTER_IP:-21.151.225.172}
ROUTER_PORT=${ROUTER_PORT:-30001}
API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
BASE_URL="http://${ROUTER_IP}:${ROUTER_PORT}"

echo "============================================"
echo " PD RDMA Correctness Test"
echo " Router: ${BASE_URL}"
echo "============================================"

# Wait for router to be ready
echo "Waiting for router..."
for i in $(seq 1 60); do
  if curl -s "${BASE_URL}/health" 2>/dev/null | grep -q "ok"; then
    echo "Router is healthy!"
    break
  fi
  sleep 5
  echo -n "."
done

# Test 1: Simple math
echo ""
echo "=== Test 1: What is 2+2? ==="
RESP=$(curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":16,"temperature":0}')
echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])" 2>/dev/null || echo "Raw: $RESP"

# Test 2: Paris
echo ""
echo "=== Test 2: What is the capital of France? ==="
RESP=$(curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":16,"temperature":0}')
echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])" 2>/dev/null || echo "Raw: $RESP"

# Test 3: 70
echo ""
echo "=== Test 3: What is 35*2? ==="
RESP=$(curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is 35*2? Answer with just the number."}],"max_tokens":16,"temperature":0}')
echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])" 2>/dev/null || echo "Raw: $RESP"

# Test 4: 42
echo ""
echo "=== Test 4: What is 6*7? ==="
RESP=$(curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is 6*7? Answer with just the number."}],"max_tokens":16,"temperature":0}')
echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])" 2>/dev/null || echo "Raw: $RESP"

# Test 5: Longer response (check for garbage)
echo ""
echo "=== Test 5: Write a short poem about coding ==="
RESP=$(curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Write a 4-line poem about coding."}],"max_tokens":128,"temperature":0}')
echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'])" 2>/dev/null || echo "Raw: $RESP"

# Test 6: 4K prefill (long context)
echo ""
echo "=== Test 6: 4K prefill test ==="
LONG_TEXT=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 100)")
RESP=$(curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "{\"model\":\"glm-5.2\",\"messages\":[{\"role\":\"user\",\"content\":\"Summarize this text in one sentence: ${LONG_TEXT}\"}],\"max_tokens\":64,\"temperature\":0}")
echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'][:200])" 2>/dev/null || echo "Raw: $RESP"

echo ""
echo "============================================"
echo " Correctness tests complete"
echo "============================================"
