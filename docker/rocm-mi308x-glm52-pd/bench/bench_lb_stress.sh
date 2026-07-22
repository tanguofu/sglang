#!/bin/bash
# Stress test: 80 concurrent identical requests to observe cache_aware routing under load
set -uo pipefail

API_URL="http://glm52-2tp8.jmpti.woa.com/v1/chat/completions"
API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL="glm-5.2"

echo "=== Load Balancing Stress Test ==="
echo "80 concurrent identical requests (same prompt, max_tokens=30)"
echo ""

echo "--- BEFORE ---"
kubectl exec -n kube-system sglang-glm52-2tp8-router-55777fb66b-txtc8 -- curl -s http://localhost:29000/metrics 2>&1 | grep "smg_worker_cb_outcomes_total" | grep "outcome=\"success\""

PROMPT="What is 2+2? Answer with only the number."
echo ""
echo "Sending 80 concurrent requests..."

for i in $(seq 1 80); do
    curl -s -o /dev/null \
        -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":30,\"temperature\":0}" &
done
wait

sleep 2
echo ""
echo "--- AFTER ---"
kubectl exec -n kube-system sglang-glm52-2tp8-router-55777fb66b-txtc8 -- curl -s http://localhost:29000/metrics 2>&1 | grep "smg_worker_cb_outcomes_total" | grep "outcome=\"success\""

echo ""
echo "--- Pod-level request metrics ---"
echo "Pod-0 (.152):"
kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- curl -s http://localhost:30000/metrics 2>&1 | grep "sglang:num_requests_total\|sglang:gen_throughput\|sglang:kv_cache_used" | head -5
echo "Pod-1 (.172):"
kubectl exec -n kube-system sglang-glm52-2tp8-sglang-1 -- curl -s http://localhost:30000/metrics 2>&1 | grep "sglang:num_requests_total\|sglang:gen_throughput\|sglang:kv_cache_used" | head -5

echo ""
echo "=== Stress test complete ==="
