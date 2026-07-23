#!/bin/bash
# Verify previous_response_id chaining works when prior response is created worker-direct
set -e

# Step 1: create prior response via worker-direct (since gateway non-stream is broken)
WORKER_RESP=$(kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- /usr/bin/curl -sS --max-time 30 \
  http://127.0.0.1:30000/v1/responses \
  -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2[1M]","input":"my name is bob","max_output_tokens":30,"store":true}')

echo "Worker-direct prior response (non-stream):"
echo "$WORKER_RESP" | /usr/bin/python3 -c "import sys,json; r=json.load(sys.stdin); print('id:', r.get('id'), 'status:', r.get('status')); print('output:', json.dumps(r.get('output',[]), ensure_ascii=False)[:400])"
RESP_ID=$(echo "$WORKER_RESP" | /usr/bin/python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))")
echo ""
echo "Response ID for chaining: $RESP_ID"

if [ -z "$RESP_ID" ]; then
  echo "FAIL: no response id"
  exit 1
fi

echo ""
echo "Step 2: streaming /v1/responses via gateway with previous_response_id=$RESP_ID"
/usr/bin/curl -sS -N --max-time 60 \
  "https://glm52-2tp8.jmpti.woa.com/v1/responses" \
  -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"glm-5.2[1M]\",\"input\":\"what is my name? one word\",\"previous_response_id\":\"$RESP_ID\",\"max_output_tokens\":30,\"stream\":true}" 2>&1 | grep -E "^event:" | head -30
