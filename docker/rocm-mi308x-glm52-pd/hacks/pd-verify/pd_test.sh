#!/bin/bash
# Comprehensive PD deployment verification test
# Tests: health, chat, responses, messages, quality, concurrency, sustained load
# Usage: bash /tmp/pd_test.sh <router_pod_name> <router_host:port>

set -uo pipefail

ROUTER_POD="${1:-sglang-ts4-1p1d-router-7697dc698-4qw7q}"
ROUTER_HOST="${2:-21.234.170.159:30001}"
API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL="glm-5.2"
RESULTS_DIR="/tmp/pd_results"
mkdir -p "$RESULTS_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Helper: send request via router pod (since we can't reach the IPs directly)
send_req() {
  local endpoint="$1"
  local payload="$2"
  local extra_headers="${3:-}"
  kubectl exec -n kube-system "$ROUTER_POD" -- bash -c "curl -s -w '\n---HTTP_CODE=%{http_code} TIME=%{time_total}\n' --max-time 120 http://$ROUTER_HOST$endpoint -H 'Content-Type: application/json' -H 'Authorization: Bearer $API_KEY' $extra_headers -d '$payload'"
}

# ============================================================
# Phase 1: Health checks
# ============================================================
log "=== Phase 1: Health checks ==="
for svc in "prefill:21.234.170.159:30000" "decode:21.234.171.87:30000" "router:$ROUTER_HOST"; do
  name="${svc%%:*}"
  addr="${svc#*:}"
  code=$(kubectl exec -n kube-system "$ROUTER_POD" -- curl -s -o /dev/null -w "%{http_code}" --max-time 10 "http://$addr/health" 2>/dev/null || echo "000")
  log "  $name ($addr): $code"
done

# ============================================================
# Phase 2: Chat completions smoke test
# ============================================================
log "=== Phase 2: /v1/chat/completions smoke test ==="
CHAT_PAYLOAD='{"model":"glm-5.2","messages":[{"role":"user","content":"Hello, introduce yourself in one sentence."}],"max_tokens":64,"temperature":0.7}'
send_req "/v1/chat/completions" "$CHAT_PAYLOAD" | tee "$RESULTS_DIR/phase2_chat.txt"
echo ""

# ============================================================
# Phase 3: Responses API smoke test (codex compatible)
# ============================================================
log "=== Phase 3: /v1/responses smoke test ==="
RESP_PAYLOAD='{"model":"glm-5.2","input":"Hello, introduce yourself in one sentence.","max_output_tokens":64,"stream":false}'
send_req "/v1/responses" "$RESP_PAYLOAD" | tee "$RESULTS_DIR/phase3_responses.txt"
echo ""

# Phase 3b: streaming responses
log "=== Phase 3b: /v1/responses streaming ==="
RESP_STREAM_PAYLOAD='{"model":"glm-5.2","input":"Count from 1 to 5.","max_output_tokens":64,"stream":true}'
send_req "/v1/responses" "$RESP_STREAM_PAYLOAD" | head -30 | tee "$RESULTS_DIR/phase3b_responses_stream.txt"
echo ""

# ============================================================
# Phase 4: Anthropic Messages API smoke test
# ============================================================
log "=== Phase 4: /v1/messages smoke test ==="
MSG_PAYLOAD='{"model":"glm-5.2","max_tokens":64,"messages":[{"role":"user","content":"Hello, introduce yourself in one sentence."}]}'
kubectl exec -n kube-system "$ROUTER_POD" -- bash -c "curl -s -w '\n---HTTP_CODE=%{http_code} TIME=%{time_total}\n' --max-time 120 http://$ROUTER_HOST/v1/messages -H 'Content-Type: application/json' -H 'x-api-key: $API_KEY' -H 'anthropic-version: 2023-06-01' -d '$MSG_PAYLOAD'" | tee "$RESULTS_DIR/phase4_messages.txt"
echo ""

# ============================================================
# Phase 5: Quality test with 10 diverse prompts
# ============================================================
log "=== Phase 5: Quality test (10 diverse prompts) ==="
PROMPTS=(
  "What is 17 * 23? Think step by step."
  "Write a Python function to check if a string is a palindrome."
  "Explain the difference between TCP and UDP in networking."
  "用中文解释什么是分布式系统的一致性。"
  "Translate to English: 今天天气很好，适合出去散步。"
  "What are the main causes of climate change? List 3."
  "Write a SQL query to find the second highest salary in a table."
  "Explain Big-O notation with an example."
  "What is the capital of France? Answer in one word."
  "Write a haiku about programming."
)
for i in "${!PROMPTS[@]}"; do
  log "  Q$i: ${PROMPTS[$i]:0:50}..."
  PAYLOAD=$(printf '{"model":"glm-5.2","messages":[{"role":"user","content":%s}],"max_tokens":128,"temperature":0.3}' "$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "${PROMPTS[$i]}")")
  echo "--- Q$i ---" >> "$RESULTS_DIR/phase5_quality.txt"
  send_req "/v1/chat/completions" "$PAYLOAD" >> "$RESULTS_DIR/phase5_quality.txt" 2>&1
  echo "" >> "$RESULTS_DIR/phase5_quality.txt"
done
log "  Quality results saved to $RESULTS_DIR/phase5_quality.txt"

# ============================================================
# Phase 6: Concurrency staircase (1, 4, 8, 16, 32)
# ============================================================
log "=== Phase 6: Concurrency staircase on /v1/responses ==="
for CONC in 1 4 8 16 32; do
  log "  Testing concurrency=$CONC (60s)..."
  DURATION=60
  START=$(date +%s)
  END=$((START + DURATION))
  PIDS=()
  OUT_FILE="$RESULTS_DIR/phase6_conc${CONC}.txt"
  > "$OUT_FILE"
  REQ_COUNT=0
  while [ $(date +%s) -lt $END ]; do
    for i in $(seq 1 $CONC); do
      PAYLOAD='{"model":"glm-5.2","input":"Write a short paragraph about artificial intelligence.","max_output_tokens":128,"stream":false}'
      kubectl exec -n kube-system "$ROUTER_POD" -- bash -c "curl -s -w '\n---HTTP=%{http_code} TIME=%{time_total}\n' --max-time 120 http://$ROUTER_HOST/v1/responses -H 'Content-Type: application/json' -H 'Authorization: Bearer $API_KEY' -d '$PAYLOAD'" >> "$OUT_FILE" 2>&1 &
      PIDS+=($!)
      REQ_COUNT=$((REQ_COUNT + 1))
    done
    # Wait for this batch to complete
    for pid in "${PIDS[@]}"; do
      wait $pid 2>/dev/null
    done
    PIDS=()
  done
  # Stats
  TOTAL=$(grep -c "^---HTTP=" "$OUT_FILE" 2>/dev/null || echo 0)
  SUCCESS=$(grep -c "^---HTTP=200" "$OUT_FILE" 2>/dev/null || echo 0)
  ERRORS=$((TOTAL - SUCCESS))
  AVG_TIME=$(grep "^---HTTP=" "$OUT_FILE" | sed 's/.*TIME=\([0-9.]*\)/\1/' | awk '{sum+=$1; count++} END {if(count>0) printf "%.3f", sum/count; else print "N/A"}')
  log "  conc=$CONC: total=$TOTAL success=$SUCCESS errors=$ERRORS avg_time=${AVG_TIME}s"
  echo "concurrency=$CONC total=$TOTAL success=$SUCCESS errors=$ERRORS avg_time=${AVG_TIME}s" >> "$RESULTS_DIR/phase6_summary.txt"
  sleep 5
done
log "  Staircase results saved to $RESULTS_DIR/phase6_summary.txt"

# ============================================================
# Phase 7: Sustained 32-concurrent for 300s
# ============================================================
log "=== Phase 7: Sustained 32-concurrent for 300s ==="
CONC=32
DURATION=300
START=$(date +%s)
END=$((START + DURATION))
OUT_FILE="$RESULTS_DIR/phase7_sustained.txt"
> "$OUT_FILE"
REQ_COUNT=0
while [ $(date +%s) -lt $END ]; do
  PIDS=()
  for i in $(seq 1 $CONC); do
    PAYLOAD='{"model":"glm-5.2","input":"Explain the concept of recursion in programming with an example.","max_output_tokens":128,"stream":false}'
    kubectl exec -n kube-system "$ROUTER_POD" -- bash -c "curl -s -w '\n---HTTP=%{http_code} TIME=%{time_total}\n' --max-time 120 http://$ROUTER_HOST/v1/responses -H 'Content-Type: application/json' -H 'Authorization: Bearer $API_KEY' -d '$PAYLOAD'" >> "$OUT_FILE" 2>&1 &
    PIDS+=($!)
    REQ_COUNT=$((REQ_COUNT + 1))
  done
  for pid in "${PIDS[@]}"; do
    wait $pid 2>/dev/null
  done
  ELAPSED=$(( $(date +%s) - START ))
  log "  ${ELAPSED}s elapsed, sent $REQ_COUNT requests so far"
done
TOTAL=$(grep -c "^---HTTP=" "$OUT_FILE" 2>/dev/null || echo 0)
SUCCESS=$(grep -c "^---HTTP=200" "$OUT_FILE" 2>/dev/null || echo 0)
ERRORS=$((TOTAL - SUCCESS))
AVG_TIME=$(grep "^---HTTP=" "$OUT_FILE" | sed 's/.*TIME=\([0-9.]*\)/\1/' | awk '{sum+=$1; count++} END {if(count>0) printf "%.3f", sum/count; else print "N/A"}')
P99_TIME=$(grep "^---HTTP=" "$OUT_FILE" | sed 's/.*TIME=\([0-9.]*\)/\1/' | sort -n | awk '{a[NR]=$1} END {if(NR>0) print a[int(NR*0.99)]}')
log "  Sustained 32-concurrent: total=$TOTAL success=$SUCCESS errors=$ERRORS avg=${AVG_TIME}s p99=${P99_TIME}s"
echo "sustained_32concurrent total=$TOTAL success=$SUCCESS errors=$ERRORS avg=${AVG_TIME}s p99=${P99_TIME}s" >> "$RESULTS_DIR/phase7_summary.txt"

# ============================================================
# Summary
# ============================================================
log "=== ALL TESTS COMPLETE ==="
log "Results in $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
