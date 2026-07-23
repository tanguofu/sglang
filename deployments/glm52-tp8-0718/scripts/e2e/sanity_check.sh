#!/bin/bash
# Sanity check current availability
for i in 1 2 3 4 5 6 7 8 9 10; do
  /usr/bin/curl -sS -o /tmp/sanity.txt -w "request $i: code=%{http_code}" --max-time 15 \
    "https://glm52-2tp8.jmpti.woa.com/v1/messages" \
    -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
    -H "anthropic-version: 2023-06-01" \
    -H "Content-Type: application/json" \
    -d '{"model":"glm-5.2","max_tokens":20,"stream":true,"messages":[{"role":"user","content":"hi"}]}'
  err=$(/usr/bin/grep -o "no_available_workers\|Service Unavailable" /tmp/sanity.txt 2>/dev/null)
  if [ -n "$err" ]; then
    echo -n "  ERROR=$err"
  fi
  echo ""
  /bin/sleep 1
done
