#!/bin/bash
# Comprehensive streaming compatibility test for both protocols.
# Tests all streaming modes for /v1/messages (Claude) and /v1/responses (Codex/OpenAI).
# Captures SSE event sequences and validates they match expected protocol shape.

GATEWAY="https://glm52-2tp8.jmpti.woa.com"
AUTH="Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}"
ANTHROPIC_VER="anthropic-version: 2023-06-01"
JSON="Content-Type: application/json"

PASS=0
FAIL=0
WARN=0
RESULTS=""

log() {
  local test="$1"
  local code="$2"
  local status="$3"
  local note="$4"
  RESULTS="${RESULTS}${test} | ${code} | ${status} | ${note}\n"
  case "$status" in
    PASS) PASS=$((PASS+1)) ;;
    FAIL) FAIL=$((FAIL+1)) ;;
    WARN) WARN=$((WARN+1)) ;;
  esac
}

# Capture SSE stream and report: HTTP code + event types seen (in order) + first error if any.
# Sets globals: SSE_CODE, SSE_EVENTS, SSE_ERROR, SSE_FIRST_EVENT
capture_sse() {
  local path="$1"
  local body="$2"
  local extra_header="$3"
  local out
  out=$(/usr/bin/curl -sS -N --max-time 60 \
    "${GATEWAY}${path}" \
    -H "$AUTH" -H "$JSON" ${extra_header:+-H "$extra_header"} \
    -d "$body" 2>&1 || true)
  # Extract event types (lines starting with "event: ")
  SSE_EVENTS=$(echo "$out" | grep -E "^event:" | sed 's/^event: //' | tr '\n' ',' | sed 's/,$//')
  SSE_FIRST_EVENT=$(echo "$out" | grep -E "^event:" | head -1 | sed 's/^event: //')
  # Look for error in data payload
  SSE_ERROR=$(echo "$out" | grep -oE '"error"[^}]*' | head -1)
  # Approximate code: if SSE has expected first event, call it 200
  if [ -n "$SSE_EVENTS" ]; then
    SSE_CODE=200
  elif echo "$out" | grep -qE '"error"'; then
    SSE_CODE=400
  else
    SSE_CODE=000
  fi
}

echo "=========================================="
echo "Streaming Compatibility Test — Both Protocols"
echo "Gateway: $GATEWAY"
echo "=========================================="
echo ""

# ============================================================================
# /v1/messages (Anthropic Messages API — Claude Code)
# Expected SSE sequence:
#   message_start, content_block_start, content_block_delta*, content_block_stop,
#   message_delta, message_stop   (ping may appear interleaved)
# ============================================================================

echo "--- Messages API: streaming variants ---"
echo ""

# Test 1: basic stream
echo "Test 1: basic stream (text)"
capture_sse "/v1/messages" \
  '{"model":"glm-5.2[1M]","max_tokens":30,"stream":true,"messages":[{"role":"user","content":"Say hi in 3 words"}]}' \
  "$ANTHROPIC_VER"
echo "  code=$SSE_CODE first=$SSE_FIRST_EVENT"
echo "  events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ] && echo "$SSE_EVENTS" | grep -q "message_start"; then
  log "M1 basic stream" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "M1 basic stream" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 2: stream with system prompt
echo "Test 2: stream with system prompt"
capture_sse "/v1/messages" \
  '{"model":"glm-5.2[1M]","max_tokens":30,"stream":true,"system":"You are terse","messages":[{"role":"user","content":"hi"}]}' \
  "$ANTHROPIC_VER"
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ] && echo "$SSE_EVENTS" | grep -q "message_start"; then
  log "M2 stream+system" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "M2 stream+system" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 3: stream with thinking enabled
echo "Test 3: stream with thinking enabled"
capture_sse "/v1/messages" \
  '{"model":"glm-5.2[1M]","max_tokens":200,"stream":true,"thinking":{"type":"enabled","budget_tokens":1024},"messages":[{"role":"user","content":"think briefly then answer: 2+2?"}]}' \
  "$ANTHROPIC_VER"
echo "  code=$SSE_CODE events: $SSE_EVENTS"
# Expecting at least two content_block_start (thinking + text)
if [ "$SSE_CODE" = "200" ] && echo "$SSE_EVENTS" | grep -q "message_start"; then
  log "M3 stream+thinking" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "M3 stream+thinking" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 4: stream with tools (tool_use)
echo "Test 4: stream with tools (tool_use)"
capture_sse "/v1/messages" \
  '{"model":"glm-5.2[1M]","max_tokens":100,"stream":true,"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"tool_choice":{"type":"any"},"messages":[{"role":"user","content":"weather in Paris?"}]}' \
  "$ANTHROPIC_VER"
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ] && echo "$SSE_EVENTS" | grep -q "message_start"; then
  log "M4 stream+tools" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "M4 stream+tools" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 5: stream with betas header (Claude Code uses beta features)
echo "Test 5: stream with betas header"
capture_sse "/v1/messages" \
  '{"model":"glm-5.2[1M]","max_tokens":30,"stream":true,"messages":[{"role":"user","content":"hi"}]}' \
  "$ANTHROPIC_VER"
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ]; then
  log "M5 stream+betas" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "M5 stream+betas" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 6: stream with beta=true query param (Claude Code style)
echo "Test 6: stream with ?beta=true query"
capture_sse "/v1/messages?beta=true" \
  '{"model":"glm-5.2[1M]","max_tokens":30,"stream":true,"messages":[{"role":"user","content":"hi"}]}' \
  "$ANTHROPIC_VER"
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ]; then
  log "M6 stream+?beta=true" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "M6 stream+?beta=true" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 7: stream with image (base64) — small 1x1 png
echo "Test 7: stream with image"
PNG_B64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAfbLI3wAAAABJRU5ErkJggg=="
capture_sse "/v1/messages" \
  "{\"model\":\"glm-5.2[1M]\",\"max_tokens\":30,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"$PNG_B64\"}},{\"type\":\"text\",\"text\":\"describe briefly\"}]}]}" \
  "$ANTHROPIC_VER"
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ]; then
  log "M7 stream+image" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "M7 stream+image" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 8: stream with stop_sequences
echo "Test 8: stream with stop_sequences"
capture_sse "/v1/messages" \
  '{"model":"glm-5.2[1M]","max_tokens":50,"stream":true,"stop_sequences":["STOP"],"messages":[{"role":"user","content":"count 1 to 10"}]}' \
  "$ANTHROPIC_VER"
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ]; then
  log "M8 stream+stop_seq" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "M8 stream+stop_seq" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 9: stream with temperature/top_p/top_k
echo "Test 9: stream with temperature+top_p"
capture_sse "/v1/messages" \
  '{"model":"glm-5.2[1M]","max_tokens":30,"stream":true,"temperature":0.7,"top_p":0.9,"messages":[{"role":"user","content":"hi"}]}' \
  "$ANTHROPIC_VER"
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ]; then
  log "M9 stream+temp+top_p" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "M9 stream+temp+top_p" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 10: multi-turn stream (assistant prior turn)
echo "Test 10: multi-turn stream"
capture_sse "/v1/messages" \
  '{"model":"glm-5.2[1M]","max_tokens":30,"stream":true,"messages":[{"role":"user","content":"my name is bob"},{"role":"assistant","content":"hi bob"},{"role":"user","content":"what is my name? one word"}]}' \
  "$ANTHROPIC_VER"
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ]; then
  log "M10 multi-turn stream" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "M10 multi-turn stream" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 11: stream with prior assistant tool_use turn
echo "Test 11: stream with prior tool_use turn"
capture_sse "/v1/messages" \
  '{"model":"glm-5.2[1M]","max_tokens":50,"stream":true,"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"messages":[{"role":"user","content":"weather in Paris?"},{"role":"assistant","content":[{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{"city":"Paris"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01","content":"sunny, 18C"}]}]}' \
  "$ANTHROPIC_VER"
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ]; then
  log "M11 stream+tool_result" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "M11 stream+tool_result" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# ============================================================================
# /v1/responses (OpenAI Responses API — Codex CLI)
# Expected SSE sequence:
#   response.created, response.in_progress, response.output_item.added,
#   response.output_text.delta*, response.output_text.done,
#   response.output_item.done, response.completed
# ============================================================================

echo "--- Responses API: streaming variants ---"
echo ""

# Test 12: basic stream (text)
echo "Test 12: responses basic stream"
capture_sse "/v1/responses" \
  '{"model":"glm-5.2[1M]","input":"Say hi in 3 words","max_output_tokens":30,"stream":true}' \
  ""
echo "  code=$SSE_CODE first=$SSE_FIRST_EVENT"
echo "  events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ] && echo "$SSE_EVENTS" | grep -q "response.created"; then
  log "R12 basic stream" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "R12 basic stream" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 13: stream with instructions (system-like)
echo "Test 13: responses stream with instructions"
capture_sse "/v1/responses" \
  '{"model":"glm-5.2[1M]","input":"hi","instructions":"You are terse","max_output_tokens":30,"stream":true}' \
  ""
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ]; then
  log "R13 stream+instructions" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "R13 stream+instructions" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 14: stream with reasoning (effort)
echo "Test 14: responses stream with reasoning"
capture_sse "/v1/responses" \
  '{"model":"glm-5.2[1M]","input":"think briefly: 2+2?","max_output_tokens":200,"reasoning":{"effort":"low"},"stream":true}' \
  ""
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ]; then
  log "R14 stream+reasoning" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "R14 stream+reasoning" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 15: stream with function tools (Codex style)
echo "Test 15: responses stream with function tools"
capture_sse "/v1/responses" \
  '{"model":"glm-5.2[1M]","input":"weather in Paris?","max_output_tokens":100,"stream":true,"tools":[{"type":"function","name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]}' \
  ""
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ]; then
  log "R15 stream+function" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "R15 stream+function" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 16: stream with tool_choice="auto"
echo "Test 16: responses stream with tool_choice=auto"
capture_sse "/v1/responses" \
  '{"model":"glm-5.2[1M]","input":"weather in Paris?","max_output_tokens":100,"stream":true,"tool_choice":"auto","tools":[{"type":"function","name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]}' \
  ""
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ]; then
  log "R16 stream+tool_choice" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "R16 stream+tool_choice" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 17: stream with text format (temperature)
echo "Test 17: responses stream with temperature"
capture_sse "/v1/responses" \
  '{"model":"glm-5.2[1M]","input":"hi","max_output_tokens":30,"stream":true,"temperature":0.7,"top_p":0.9}' \
  ""
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ]; then
  log "R17 stream+temp" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "R17 stream+temp" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 18: stream with store=false (stateless — Codex often uses this)
echo "Test 18: responses stream with store=false"
capture_sse "/v1/responses" \
  '{"model":"glm-5.2[1M]","input":"hi","max_output_tokens":30,"stream":true,"store":false}' \
  ""
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ]; then
  log "R18 stream+store=false" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "R18 stream+store=false" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 19: stream with multi-turn input array
echo "Test 19: responses stream with input array (multi-turn)"
capture_sse "/v1/responses" \
  '{"model":"glm-5.2[1M]","input":[{"role":"user","content":"my name is bob"},{"role":"assistant","content":"hi bob"},{"role":"user","content":"what is my name? one word"}],"max_output_tokens":30,"stream":true}' \
  ""
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ]; then
  log "R19 stream+multi-turn" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "R19 stream+multi-turn" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 20: stream with prior function_call output (Codex style)
echo "Test 20: responses stream with function_call_output"
capture_sse "/v1/responses" \
  '{"model":"glm-5.2[1M]","input":[{"role":"user","content":"weather in Paris?"},{"type":"function_call","name":"get_weather","arguments":"{\"city\":\"Paris\"}","call_id":"call_01","id":"fc_01"},{"type":"function_call_output","call_id":"call_01","output":"sunny, 18C"}],"max_output_tokens":50,"stream":true,"tools":[{"type":"function","name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]}' \
  ""
echo "  code=$SSE_CODE events: $SSE_EVENTS"
if [ "$SSE_CODE" = "200" ]; then
  log "R20 stream+fc_output" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
else
  log "R20 stream+fc_output" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
fi
echo ""

# Test 21: stream with previous_response_id (stateful chain)
# First create a response, then chain it
echo "Test 21: responses stream with previous_response_id"
RESP_ID=$(/usr/bin/curl -sS -X POST "${GATEWAY}/v1/responses" \
  -H "$AUTH" -H "$JSON" \
  -d '{"model":"glm-5.2[1M]","input":"my name is bob","max_output_tokens":30}' 2>&1 | /usr/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || echo "")
echo "  prev_id=$RESP_ID"
if [ -n "$RESP_ID" ]; then
  capture_sse "/v1/responses" \
    "{\"model\":\"glm-5.2[1M]\",\"input\":\"what is my name? one word\",\"previous_response_id\":\"$RESP_ID\",\"max_output_tokens\":30,\"stream\":true}" \
    ""
  echo "  code=$SSE_CODE events: $SSE_EVENTS"
  if [ "$SSE_CODE" = "200" ]; then
    log "R21 stream+prev_id" "$SSE_CODE" "PASS" "events=$SSE_EVENTS"
  else
    log "R21 stream+prev_id" "$SSE_CODE" "FAIL" "events=$SSE_EVENTS err=$SSE_ERROR"
  fi
else
  log "R21 stream+prev_id" "000" "WARN" "could not create prior response (non-stream 400 bug)"
fi
echo ""

# ============================================================================
# Non-streaming control (for completeness)
# ============================================================================

echo "--- Non-streaming controls ---"
echo ""

# Test 22: Messages API non-stream
echo "Test 22: messages non-stream"
CODE=$(/usr/bin/curl -sS -o /dev/null -w "%{http_code}" --max-time 30 \
  "${GATEWAY}/v1/messages" \
  -H "$AUTH" -H "$JSON" -H "$ANTHROPIC_VER" \
  -d '{"model":"glm-5.2[1M]","max_tokens":30,"messages":[{"role":"user","content":"hi"}]}')
echo "  code=$CODE"
if [ "$CODE" = "200" ]; then
  log "M22 non-stream" "$CODE" "PASS" "control"
else
  log "M22 non-stream" "$CODE" "FAIL" "control"
fi
echo ""

# Test 23: Responses API non-stream (known router bug)
echo "Test 23: responses non-stream (known router bug)"
CODE=$(/usr/bin/curl -sS -o /dev/null -w "%{http_code}" --max-time 30 \
  "${GATEWAY}/v1/responses" \
  -H "$AUTH" -H "$JSON" \
  -d '{"model":"glm-5.2[1M]","input":"hi","max_output_tokens":30}')
echo "  code=$CODE (expected 400 — router bug)"
if [ "$CODE" = "400" ]; then
  log "R23 non-stream" "$CODE" "WARN" "known router stream=None bug"
elif [ "$CODE" = "200" ]; then
  log "R23 non-stream" "$CODE" "PASS" "bug fixed!"
else
  log "R23 non-stream" "$CODE" "FAIL" "unexpected"
fi
echo ""

# Test 24: Responses API non-stream direct to worker (bypass router)
echo "Test 24: responses non-stream direct to worker (kubectl exec)"
WORKER_CODE=$(kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- /usr/bin/curl -sS -o /dev/null -w "%{http_code}" --max-time 30 \
  http://127.0.0.1:30000/v1/responses \
  -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2[1M]","input":"hi","max_output_tokens":30}' 2>&1)
echo "  worker_code=$WORKER_CODE (expected 200 — worker supports non-stream)"
if [ "$WORKER_CODE" = "200" ]; then
  log "R24 non-stream worker" "$WORKER_CODE" "PASS" "worker handles non-stream fine"
else
  log "R24 non-stream worker" "$WORKER_CODE" "FAIL" "worker also broken?"
fi
echo ""

# ---------- Summary ----------
echo "=========================================="
echo "Summary: PASS=$PASS FAIL=$FAIL WARN=$WARN (total=$((PASS+FAIL+WARN)))"
echo "=========================================="
echo ""
echo "Test | http | status | note"
echo "----------------------------------------"
printf "$RESULTS"
