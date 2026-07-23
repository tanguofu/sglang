#!/bin/bash
# Test reasoning effort levels on GLM-5.2
GATEWAY="https://glm52-2tp8.jmpti.woa.com"
AUTH="Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}"
JSON="Content-Type: application/json"

for effort in "none" "low" "medium" "high"; do
  echo "=== effort=$effort ==="
  start=$(/bin/date +%s.%N)
  /usr/bin/curl -sS -N --max-time 60 \
    "${GATEWAY}/v1/responses" \
    -H "$AUTH" -H "$JSON" \
    -d "{\"model\":\"glm-5.2\",\"input\":\"What is 15*17? Just the number.\",\"max_output_tokens\":400,\"reasoning\":{\"effort\":\"$effort\"},\"stream\":true}" 2>&1 > /tmp/effort_out.txt
  end=$(/bin/date +%s.%N)
  elapsed=$(/usr/bin/awk "BEGIN{printf \"%.2f\", $end-$start}")
  events=$(/usr/bin/grep -E "^event:" /tmp/effort_out.txt | /usr/bin/sort | /usr/bin/uniq -c | /usr/bin/tr '\n' ' ')
  reason_count=$(/usr/bin/grep -c "reasoning_text.delta" /tmp/effort_out.txt)
  output_count=$(/usr/bin/grep -c "output_text.delta" /tmp/effort_out.txt)
  # Get final usage
  usage=$(/usr/bin/grep "response.completed" /tmp/effort_out.txt | /usr/bin/tail -1 | /usr/bin/python3 -c "import sys,json; d=json.loads(sys.stdin.read()[6:]); u=d['response']['usage']; print(f\"in={u['input_tokens']} out={u['output_tokens']} reason={u.get('output_tokens_details',{}).get('reasoning_tokens','?')}\")" 2>/dev/null)
  echo "  elapsed=${elapsed}s reason_delta=$reason_count output_delta=$output_count usage=$usage"
  echo "  events: $events"
done
