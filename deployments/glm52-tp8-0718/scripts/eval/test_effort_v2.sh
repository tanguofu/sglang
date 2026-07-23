#!/bin/bash
# Test effort=minimal vs low vs medium vs high on GLM-5.2 (worker-direct to avoid router non-stream bug)
GATEWAY="https://glm52-2tp8.jmpti.woa.com"
WORKER="http://127.0.0.1:30000"
AUTH="Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}"
JSON="Content-Type: application/json"

echo "Testing reasoning effort levels (streaming via gateway, same prompt)"
echo "Prompt: 'Write a Python function to check if a string is a palindrome. Just the code.'"
echo ""

for effort in "minimal" "low" "medium" "high"; do
  echo "=== effort=$effort ==="
  start=$(/bin/date +%s.%N)
  /usr/bin/curl -sS -N --max-time 120 \
    "${GATEWAY}/v1/responses" \
    -H "$AUTH" -H "$JSON" \
    -d "{\"model\":\"glm-5.2\",\"input\":\"Write a Python function to check if a string is a palindrome. Just the code, no explanation.\",\"max_output_tokens\":600,\"reasoning\":{\"effort\":\"$effort\"},\"stream\":true}" > /tmp/effort_$effort.txt 2>&1
  end=$(/bin/date +%s.%N)
  elapsed=$(/usr/bin/awk "BEGIN{printf \"%.2f\", $end-$start}")
  reason_count=$(/usr/bin/grep -c "reasoning_text.delta" /tmp/effort_$effort.txt)
  output_count=$(/usr/bin/grep -c "output_text.delta" /tmp/effort_$effort.txt)
  # Get final usage from response.completed
  usage=$(/usr/bin/grep "response.completed" /tmp/effort_$effort.txt | /usr/bin/python3 -c "
import sys, json
for line in sys.stdin:
    if line.startswith('data: '):
        d = json.loads(line[6:])
        u = d['response']['usage']
        print(f\"in={u['input_tokens']} out={u['output_tokens']} total={u['total_tokens']}\")
        break
" 2>/dev/null)
  echo "  elapsed=${elapsed}s  reason_deltas=$reason_count  output_deltas=$output_count  usage=$usage"
done
