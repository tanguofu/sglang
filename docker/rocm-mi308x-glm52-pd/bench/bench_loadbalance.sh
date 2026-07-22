#!/bin/bash
# Load balancing test — send identical prompts and track which worker handles each
# Uses a unique prefix in each batch to observe cache_aware routing behavior.
set -uo pipefail

API_URL="http://glm52-2tp8.jmpti.woa.com/v1/chat/completions"
API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL="glm-5.2"

echo "=== Load Balancing Test ==="
echo ""
echo "Strategy: 3 batches of 20 identical requests each, different prompt per batch."
echo "Expected: cache_aware should route 1st request to either worker, then"
echo "subsequent identical-prompt requests to the same worker (prefix cache hit)."
echo ""

# Capture worker metrics before
echo "--- Worker request counts BEFORE test ---"
kubectl exec -n kube-system sglang-glm52-2tp8-router-55777fb66b-txtc8 -- curl -s http://localhost:29000/metrics 2>&1 | grep "smg_worker_cb_outcomes_total" | grep "outcome=\"success\""
echo ""

BATCH1_PROMPT="What is the capital of Japan? Answer in one word."
BATCH2_PROMPT="Explain how HTTPS encryption works in 2 sentences."
BATCH3_PROMPT="What is the boiling point of water in Celsius? Number only."

run_batch() {
    local batch_name=$1
    local prompt=$2
    local count=$3

    echo "--- $batch_name: $count identical requests ---"
    echo "Prompt: ${prompt:0:60}..."

    local tmpdir=$(mktemp -d)

    for i in $(seq 1 $count); do
        curl -s -o /dev/null \
            -X POST "$API_URL" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $API_KEY" \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":50,\"temperature\":0}" &
    done
    wait

    rm -rf "$tmpdir"
    echo "Done. Checking worker metrics..."
    sleep 1

    kubectl exec -n kube-system sglang-glm52-2tp8-router-55777fb66b-txtc8 -- curl -s http://localhost:29000/metrics 2>&1 | grep "smg_worker_cb_outcomes_total" | grep "outcome=\"success\""
    echo ""
}

run_batch "Batch 1 (unique prompt A)" "$BATCH1_PROMPT" 20
run_batch "Batch 2 (unique prompt B)" "$BATCH2_PROMPT" 20
run_batch "Batch 3 (unique prompt C)" "$BATCH3_PROMPT" 20

echo "--- Final worker metrics ---"
kubectl exec -n kube-system sglang-glm52-2tp8-router-55777fb66b-txtc8 -- curl -s http://localhost:29000/metrics 2>&1 | grep -E "smg_worker_cb_outcomes_total|smg_worker_requests_active|smg_worker_health" | head -10

echo ""
echo "--- Worker KV cache metrics ---"
echo "Pod-0 (.152):"
kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- curl -s http://localhost:30000/metrics 2>&1 | grep -E "sglang:kv_cache_used|sglang:hicache_host_used" | head -4
echo "Pod-1 (.172):"
kubectl exec -n kube-system sglang-glm52-2tp8-sglang-1 -- curl -s http://localhost:30000/metrics 2>&1 | grep -E "sglang:kv_cache_used|sglang:hicache_host_used" | head -4

echo ""
echo "=== Load balancing test complete ==="
