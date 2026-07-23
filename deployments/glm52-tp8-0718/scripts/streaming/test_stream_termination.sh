#!/bin/bash
# Test stream termination patterns and include_usage for all protocols.

GATEWAY="https://glm52-2tp8.jmpti.woa.com"
AUTH="Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}"
ANTHROPIC_VER="anthropic-version: 2023-06-01"
JSON="Content-Type: application/json"

echo "=== /v1/chat/completions with stream_options.include_usage ==="
/usr/bin/curl -sS -N --max-time 60 \
  "${GATEWAY}/v1/chat/completions" \
  -H "$AUTH" -H "$JSON" \
  -d '{"model":"glm-5.2[1M]","max_tokens":15,"stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}' 2>&1 | grep -E "^(event:|data:)" | tail -10
echo ""

echo "=== /v1/messages stream termination (look for message_stop + [DONE]) ==="
/usr/bin/curl -sS -N --max-time 60 \
  "${GATEWAY}/v1/messages" \
  -H "$AUTH" -H "$ANTHROPIC_VER" -H "$JSON" \
  -d '{"model":"glm-5.2[1M]","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}' 2>&1 | tail -8
echo ""

echo "=== /v1/responses stream termination (look for response.completed + [DONE]) ==="
/usr/bin/curl -sS -N --max-time 60 \
  "${GATEWAY}/v1/responses" \
  -H "$AUTH" -H "$JSON" \
  -d '{"model":"glm-5.2[1M]","input":"hi","max_output_tokens":10,"stream":true}' 2>&1 | tail -8
