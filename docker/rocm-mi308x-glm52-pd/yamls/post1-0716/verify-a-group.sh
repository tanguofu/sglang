#!/usr/bin/env bash
# A-group PD correctness verification (post1-0716 image)
# Tests via router (pd-router-144, port 30002):
#   1. Correctness: 2+2, Paris, 42, poem (no garbage)
#   2. PD metrics: prefill #bootstrap-req>0, decode #transfer-req>0 (KV migration happens)
#   3. MTP: decode accept_len 2-3
set -euo pipefail

ROUTER_IP=${ROUTER_IP:-21.151.225.144}
ROUTER_PORT=${ROUTER_PORT:-30002}
PREFILL_IP=${PREFILL_IP:-21.151.225.144}
DECODE_IP=${DECODE_IP:-21.151.225.132}
API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
BASE_URL="http://${ROUTER_IP}:${ROUTER_PORT}"

echo "============================================"
echo " A-group PD Correctness Test (post1-0716)"
echo " Router: ${BASE_URL}"
echo "============================================"

# Wait for router
echo "Waiting for router..."
for i in $(seq 1 60); do
  if curl -s "${BASE_URL}/health" 2>/dev/null | grep -q "ok"; then
    echo "Router healthy!"; break
  fi
  sleep 5; echo -n "."
done

# --- Correctness tests ---
test_q() {
  local q="$1" expect="$2"
  local resp=$(curl -s "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" -H "Authorization: Bearer ${API_KEY}" \
    -d "{\"model\":\"glm-5.2\",\"messages\":[{\"role\":\"user\",\"content\":\"${q}\"}],\"max_tokens\":16,\"temperature\":0}")
  local ans=$(echo "$resp" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "PARSE_FAIL: $resp")
  echo "Q: ${q}  =>  ${ans}"
}

echo ""; echo "=== Correctness ==="
test_q "What is 2+2? Answer with just the number." "4"
test_q "What is the capital of France? Answer in one word." "Paris"
test_q "What is 6*7? Answer with just the number." "42"

echo ""; echo "=== Garbage check (poem) ==="
RESP=$(curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" -H "Authorization: Bearer ${API_KEY}" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Write a 4-line poem about coding."}],"max_tokens":128,"temperature":0}')
echo "$RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "RAW: $RESP"

# --- PD metrics: bootstrap-req (prefill) + transfer-req (decode) ---
echo ""; echo "=== PD KV migration metrics ==="
echo "--- Prefill (.144) bootstrap-req ---"
curl -s "http://${PREFILL_IP}:30000/metrics" 2>/dev/null | grep -E "bootstrap_req|disaggregation" | grep -v "^#" | head -5
echo "--- Decode (.132) transfer-req + accept_len ---"
curl -s "http://${DECODE_IP}:30000/metrics" 2>/dev/null | grep -E "transfer_req|accept_len|speculative" | grep -v "^#" | head -8

echo ""
echo "============================================"
echo " Verification complete"
echo " Check: bootstrap-req>0 (prefill) + transfer-req>0 (decode=KV migrated)"
echo " Check: accept_len 2-3 (MTP working)"
echo " Check: answers correct + no garbage"
echo "============================================"
