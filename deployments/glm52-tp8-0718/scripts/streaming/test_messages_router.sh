#!/bin/bash
ROUTER_POD=$(kubectl get pods -n kube-system -l app=sglang-router -o jsonpath='{.items[0].metadata.name}')
echo "Router pod: $ROUTER_POD"
echo ""
for path in "/v1/messages" "/v1/responses" "/v1/chat/completions" "/health"; do
  code=$(kubectl exec -n kube-system $ROUTER_POD -- curl -sS -o /dev/null -w "%{http_code}" --max-time 10 \
    -X POST http://127.0.0.1:30001${path} \
    -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
    -H "anthropic-version: 2023-06-01" \
    -H "Content-Type: application/json" \
    -d '{"model":"glm-5.2[1M]","max_tokens":5,"messages":[{"role":"user","content":"hi"}],"input":"hi","max_output_tokens":5}' 2>&1)
  echo "POST $path -> $code"
done
echo ""
echo "=== Full /v1/messages response ==="
kubectl exec -n kube-system $ROUTER_POD -- curl -sS --max-time 30 -w "\nhttp_code: %{http_code}\n" \
  http://127.0.0.1:30001/v1/messages \
  -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2[1M]","max_tokens":30,"messages":[{"role":"user","content":"Say hello in 5 words"}]}' 2>&1 | tail -5
