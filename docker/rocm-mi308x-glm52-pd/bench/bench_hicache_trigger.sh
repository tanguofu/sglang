#!/bin/bash
# Trigger HiCache by filling GPU KV cache, then observe host cache usage
set -uo pipefail

API_URL="http://glm52-2tp8.jmpti.woa.com/v1/chat/completions"
API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL="glm-5.2"

echo "=== HiCache Trigger Test ==="
echo ""
echo "GPU KV cache capacity: ~509K tokens per pod"
echo "Strategy: send 20 requests with ~4K unique prompt each = ~80K tokens total,"
echo "then send 20 more with different ~4K prompts = ~160K total, pushing towards capacity."
echo ""

# Generate a long unique prompt (~4K tokens each)
gen_long_prompt() {
    local idx=$1
    # Each prompt ~4000 chars ≈ ~1000 tokens. Use unique content per request.
    python3 -c "
import random, string
random.seed($idx)
words = ['system','design','architecture','scalability','latency','throughput',
         'reliability','consistency','availability','partition','tolerance',
         'replication','sharding','indexing','caching','queueing','batching',
         'streaming','concurrency','parallelism','asynchronous','synchronous',
         'distributed','centralized','decentralized','federated','hybrid',
         'microservice','monolith','serverless','container','orchestration']
lines = []
for i in range(400):
    line = ' '.join(random.sample(words, 12))
    lines.append(f'{i}. {line}')
print('Analyze the following system design scenario in detail: ' + ' '.join(lines))
"
}

echo "--- BEFORE: HiCache metrics ---"
kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- curl -s http://localhost:30000/metrics 2>&1 | grep -E "hicache_host_used|kv_used_tokens|kv_available_tokens|kv_evictable_tokens|cached_tokens_total" | grep -v "^#" | head -10
echo ""

echo "Sending 40 long-prompt requests sequentially (fill GPU KV cache)..."
for i in $(seq 1 40); do
    prompt=$(gen_long_prompt $i)
    curl -s -o /dev/null \
        -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "$(python3 -c "
import json
prompt = '''$prompt'''
print(json.dumps({
    'model': 'glm-5.2',
    'messages': [{'role': 'user', 'content': prompt}],
    'max_tokens': 10,
    'temperature': 0
}))
")" 2>&1
    if (( i % 10 == 0 )); then
        echo "  Completed $i/40 requests"
    fi
done

echo ""
echo "--- AFTER: HiCache metrics ---"
echo "Pod-0 (.152):"
kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- curl -s http://localhost:30000/metrics 2>&1 | grep -E "hicache_host_used|kv_used_tokens|kv_available_tokens|kv_evictable_tokens|cached_tokens_total|cache_hit_rate" | grep -v "^#" | head -10
echo ""
echo "Pod-1 (.172):"
kubectl exec -n kube-system sglang-glm52-2tp8-sglang-1 -- curl -s http://localhost:30000/metrics 2>&1 | grep -E "hicache_host_used|kv_used_tokens|kv_available_tokens|kv_evictable_tokens|cached_tokens_total|cache_hit_rate" | grep -v "^#" | head -10

echo ""
echo "=== HiCache trigger test complete ==="
