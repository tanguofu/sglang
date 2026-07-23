#!/bin/bash
for path in "/v1/messages" "/v1/responses" "/v1/chat/completions" "/health" "/"; do
  code=$(kubectl exec -n kube-system sglang-glm52-2tp8-router-6b9594c895-g29ch -- curl -sS -o /dev/null -w "%{http_code}" --max-time 10 \
    -X POST http://127.0.0.1:30001${path} \
    -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"model":"glm-5.2[1M]","max_tokens":5,"messages":[{"role":"user","content":"hi"}],"input":"hi","max_output_tokens":5}' 2>&1)
  echo "POST $path -> $code"
done
