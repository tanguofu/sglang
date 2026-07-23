#!/bin/bash
# Test which model names server accepts
GATEWAY="https://glm52-2tp8.jmpti.woa.com"
AUTH="Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}"
ANTH="anthropic-version: 2023-06-01"
JSON="Content-Type: application/json"

for model in "glm-5.2[1M]" "glm-5.2" "unknown-model"; do
  code=$(/usr/bin/curl -sS -o /dev/null -w "%{http_code}" --max-time 30 \
    "${GATEWAY}/v1/messages" \
    -H "$AUTH" -H "$ANTH" -H "$JSON" \
    -d "{\"model\":\"${model}\",\"max_tokens\":5,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")
  echo "model=${model} -> ${code}"
done
