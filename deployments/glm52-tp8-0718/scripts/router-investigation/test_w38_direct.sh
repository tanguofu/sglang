#!/bin/bash
# Test .38 worker directly with a real generation request
echo "=== Direct to .38 worker ==="
kubectl exec -n kube-system sglang-glm52-2tp8-router-697df7c955-2zrcc -- curl -s http://21.234.170.38:30000/v1/chat/completions \
  -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"say hi"}],"max_tokens":20,"stream":false}' \
  -m 60 -w "\nHTTP:%{http_code} t:%{time_total}s\n" 2>&1 | tail -15

echo ""
echo "=== Direct to .103 worker ==="
kubectl exec -n kube-system sglang-glm52-2tp8-router-697df7c955-2zrcc -- curl -s http://21.234.170.103:30000/v1/chat/completions \
  -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"say hi"}],"max_tokens":20,"stream":false}' \
  -m 60 -w "\nHTTP:%{http_code} t:%{time_total}s\n" 2>&1 | tail -15

echo ""
echo "=== Check .38 worker logs for request handling ==="
kubectl logs -n kube-system sglang-glm52-2tp8-w2-sglang-0 --tail=200 2>&1 | grep -vE "NCCL|adjustment|Channels|minNChannels" | tail -20
