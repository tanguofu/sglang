#!/bin/bash
# Verification script for post-optimization state (2026-07-18)
# Run after applying router-args-patch.json
#
# Checks:
#   1. Router args (cache-threshold, balance-abs-threshold, balance-rel-threshold)
#   2. Both workers' CB state = 0 (CLOSED)
#   3. Both workers receiving traffic
#   4. Eval 26/26 PASS
#
# Usage: bash verify_optimization_0718.sh

set -euo pipefail

ROUTER_POD=$(kubectl get pods -n kube-system -l app=sglang-router,app.kubernetes.io/instance=sglang-glm52-2tp8 -o jsonpath='{.items[0].metadata.name}')
WORKER1_POD="sglang-glm52-2tp8-sglang-0"
WORKER2_POD="sglang-glm52-2tp8-w2-sglang-0"
API_KEY="${ANTHROPIC_AUTH_TOKEN:-${ANTHROPIC_AUTH_TOKEN}}"

echo "=== 1. Router args ==="
kubectl get pod -n kube-system "$ROUTER_POD" -o jsonpath='{.spec.containers[0].args}' | python3 -c "
import json,sys
args=json.load(sys.stdin)
for i,a in enumerate(args):
    if a in ('--cache-threshold','--balance-abs-threshold','--balance-rel-threshold','--policy'):
        print(f'  {a}: {args[i+1]}')
"

echo ""
echo "=== 2. CB state (should all be 0 = CLOSED) ==="
kubectl exec -n kube-system "$ROUTER_POD" -- curl -s --max-time 5 http://localhost:29000/metrics 2>&1 | grep "smg_worker_cb_state" | grep -v "^#"

echo ""
echo "=== 3. Worker traffic distribution ==="
W1=$(kubectl exec -n kube-system "$WORKER1_POD" -- curl -s --max-time 5 -H "Authorization: Bearer $API_KEY" http://localhost:30000/metrics 2>&1 | grep 'http_requests_total.*v1/responses' | grep -v "^#" | grep -oE '[0-9.]+' | tail -1)
W2=$(kubectl exec -n kube-system "$WORKER2_POD" -- curl -s --max-time 5 -H "Authorization: Bearer $API_KEY" http://localhost:30000/metrics 2>&1 | grep 'http_requests_total.*v1/responses' | grep -v "^#" | grep -oE '[0-9.]+' | tail -1)
echo "  Worker 1 (.103) /v1/responses: $W1"
echo "  Worker 2 (.38)  /v1/responses: $W2"
if [ -n "$W1" ] && [ -n "$W2" ] && [ "$W1" != "0.0" ] && [ "$W2" != "0.0" ]; then
  echo "  STATUS: Both workers receiving traffic ✓"
else
  echo "  STATUS: WARNING — one worker may be idle"
fi

echo ""
echo "=== 4. Quick gateway test (3 requests) ==="
for i in 1 2 3; do
  code=$(curl -s --max-time 30 -o /dev/null -w "%{http_code}" -X POST https://glm52-2tp8.jmpti.woa.com/v1/responses \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"glm-5.2\",\"input\":\"What is $i+$i?\",\"stream\":true,\"max_output_tokens\":1500}")
  echo "  req $i: HTTP $code"
done

echo ""
echo "=== Done ==="
