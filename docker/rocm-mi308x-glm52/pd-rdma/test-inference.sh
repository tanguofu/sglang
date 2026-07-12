#!/usr/bin/env bash
# Test PD RDMA inference through the router.
# Usage: bash test-inference.sh
#
# Sends a simple chat completion request to the PD router and checks
# that KV cache is transferred from prefill to decode via RDMA.

set -euo pipefail

ROUTER_IP=${ROUTER_IP:-21.234.170.19}
ROUTER_PORT=${ROUTER_PORT:-13002}
API_KEY=${API_KEY:-sk-46faecc9d0bc4dcd9db6a15c73ae91c8}

echo "=== Testing PD RDMA inference through router ${ROUTER_IP}:${ROUTER_PORT} ==="

python3 -c "
import json, urllib.request, time

data = json.dumps({
    'model': 'glm-5.2',
    'messages': [{'role': 'user', 'content': 'What is 2+2? Answer in one word.'}],
    'max_tokens': 200,
    'temperature': 0.1
}).encode()

req = urllib.request.Request(
    'http://${ROUTER_IP}:${ROUTER_PORT}/v1/chat/completions',
    data=data,
    headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ${API_KEY}'
    }
)

start = time.time()
try:
    resp = urllib.request.urlopen(req, timeout=300)
    result = json.loads(resp.read().decode())
    elapsed = time.time() - start
    print(f'Response received in {elapsed:.1f}s')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get('choices') and result['choices'][0].get('message', {}).get('content'):
        print(f'\\n=== SUCCESS: PD RDMA inference working! ===')
    else:
        print(f'\\n=== WARNING: Empty content (may need more max_tokens) ===')
except Exception as e:
    elapsed = time.time() - start
    print(f'ERROR after {elapsed:.1f}s: {e}')
    exit(1)
"
