#!/bin/bash
# Correctness & alignment test — same prompt to gateway, pod-0 (via kubectl), pod-1 (via kubectl)
# Saves responses to files to avoid shell quoting issues.

set -euo pipefail

API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL="glm-5.2"
GATEWAY="http://glm52-2tp8.jmpti.woa.com/v1/chat/completions"
TMPDIR=$(mktemp -d)

# Test prompts — deterministic, factual
declare -a PROMPTS=(
    "What is 17 multiplied by 23? Answer with only the number."
    "List the first 5 prime numbers in ascending order, comma-separated."
    "What is the chemical symbol for gold? Answer in one word."
    "Translate 'Good morning' to French, Spanish, and Japanese. One per line."
    "Write a Python function that returns the factorial of n using recursion. Code only."
)

echo "=== Correctness Test: 5 prompts x 3 endpoints (gateway, pod-0, pod-1) ==="
echo ""

for idx in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$idx]}"
    pnum=$((idx+1))
    echo "--- Prompt $pnum: ${prompt:0:60}... ---"

    # Gateway
    curl -s -X POST "$GATEWAY" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":100,\"temperature\":0}" \
        > "$TMPDIR/gw_$pnum.json" 2>&1

    # Pod-0 via kubectl exec (curl from inside the pod to localhost)
    kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- curl -s -X POST "http://localhost:30000/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":100,\"temperature\":0}" \
        > "$TMPDIR/pod0_$pnum.json" 2>&1

    # Pod-1 via kubectl exec
    kubectl exec -n kube-system sglang-glm52-2tp8-sglang-1 -- curl -s -X POST "http://localhost:30000/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":100,\"temperature\":0}" \
        > "$TMPDIR/pod1_$pnum.json" 2>&1

    # Parse all three with Python
    python3 -c "
import json, sys

files = {
    'gateway': '$TMPDIR/gw_$pnum.json',
    'pod-0':   '$TMPDIR/pod0_$pnum.json',
    'pod-1':   '$TMPDIR/pod1_$pnum.json',
}

for name, path in files.items():
    try:
        with open(path) as f:
            r = json.load(f)
        if 'error' in r:
            print(f'  [{name}] ERROR: {r[\"error\"]}')
            continue
        choice = r['choices'][0]
        msg = choice['message']
        content = (msg.get('content') or '').strip()
        reasoning = (msg.get('reasoning_content') or '').strip()
        usage = r.get('usage', {})
        finish = choice.get('finish_reason')
        pt = usage.get('prompt_tokens', 0)
        ct = usage.get('completion_tokens', 0)
        rt = usage.get('reasoning_tokens', 0)
        print(f'  [{name}] finish={finish} content_len={len(content)} reasoning_len={len(reasoning)} prompt_tok={pt} completion_tok={ct} reasoning_tok={rt}')
        if content:
            preview = content.replace(chr(10), ' ')[:100]
            print(f'           content: {preview}')
        else:
            # Show reasoning preview if content is empty (hit max_tokens during reasoning)
            rpreview = reasoning.replace(chr(10), ' ')[:100]
            print(f'           reasoning: {rpreview}...')
    except Exception as e:
        print(f'  [{name}] PARSE ERROR: {e}')
"
    echo ""
done

echo "=== Correctness test complete ==="
rm -rf "$TMPDIR"
