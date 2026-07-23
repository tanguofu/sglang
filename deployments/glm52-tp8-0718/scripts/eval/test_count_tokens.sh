#!/bin/bash
ROUTER_POD=$(kubectl get pods -n kube-system -l app=sglang-router -o jsonpath='{.items[0].metadata.name}')
echo "Router pod: $ROUTER_POD"
echo ""
echo "=== Test various count_tokens paths through router ==="
for path in "/v1/messages/count_tokens" "/v1/messages%2Fcount_tokens" "/messages/count_tokens"; do
  code=$(kubectl exec -n kube-system "$ROUTER_POD" -- curl -sS -o /dev/null -w "%{http_code}" --max-time 10 \
    -X POST "http://127.0.0.1:30001${path}" \
    -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
    -H "anthropic-version: 2023-06-01" \
    -H "Content-Type: application/json" \
    -d '{"model":"glm-5.2[1M]","messages":[{"role":"user","content":"hi"}]}' 2>&1)
  echo "POST $path -> $code"
done
echo ""
echo "=== Confirm worker serves count_tokens ==="
kubectl exec -n kube-system sglang-glm52-2tp8-w2-sglang-0 -- curl -sS --max-time 10 \
  http://127.0.0.1:30000/v1/messages/count_tokens \
  -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2[1M]","messages":[{"role":"user","content":"hi"}]}'
