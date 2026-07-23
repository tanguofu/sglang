#!/bin/bash
# Capture 503 events with router logs
echo "=== Baseline router log line count ==="
baseline=$(kubectl logs -n kube-system sglang-glm52-2tp8-router-697df7c955-2zrcc --tail=1 2>&1 | wc -l)
echo "Baseline: $baseline lines"

echo ""
echo "=== Sending 20 requests, capturing 503s ==="
fail_count=0
success_count=0
fail_times=""
for i in $(seq 1 20); do
  result=$(curl -s -w "\n__HTTP:%{http_code}__\n" https://glm52-2tp8.jmpti.woa.com/v1/chat/completions \
    -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is 2+2? Just the number."}],"max_tokens":1500,"stream":false}' 2>&1)
  code=$(echo "$result" | tail -1 | sed 's/.*__HTTP:\([0-9]*\)__.*/\1/')
  if [ "$code" = "503" ]; then
    fail_count=$((fail_count + 1))
    fail_times="$fail_times req$i"
    echo "req$i: 503 FAIL"
  else
    success_count=$((success_count + 1))
    echo "req$i: $code OK"
  fi
done

echo ""
echo "=== Summary: $success_count OK, $fail_count FAIL (503) ==="
echo "Failed at:$fail_times"

echo ""
echo "=== Router logs since baseline (non-health) ==="
kubectl logs -n kube-system sglang-glm52-2tp8-router-697df7c955-2zrcc --since=5m 2>&1 | grep -v "/health" | grep -v "GET " | tail -40
