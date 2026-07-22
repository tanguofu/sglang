#!/bin/bash
# Correctness test v2 — larger max_tokens to let reasoning complete
set -euo pipefail
API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL="glm-5.2"
GATEWAY="http://glm52-2tp8.jmpti.woa.com/v1/chat/completions"
TMPDIR=$(mktemp -d)

declare -a PROMPTS=(
    "List the first 5 prime numbers in ascending order, comma-separated."
    "What is the chemical symbol for gold? Answer in one word."
    "Translate 'Good morning' to French, Spanish, and Japanese. One per line."
    "Write a Python function that returns the factorial of n using recursion. Code only."
    "What is the capital of France? Answer in one word."
    "Calculate 125 + 375. Answer with only the number."
)

echo "=== Correctness Test v2: 6 prompts x 3 endpoints, max_tokens=500 ==="
echo ""

for idx in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$idx]}"
    pnum=$((idx+1))
    echo "--- Prompt $pnum: ${prompt:0:60}... ---"

    curl -s -X POST "$GATEWAY" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":500,\"temperature\":0}" \
        > "$TMPDIR/gw_$pnum.json" 2>&1

    kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- curl -s -X POST "http://localhost:30000/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":500,\"temperature\":0}" \
        > "$TMPDIR/pod0_$pnum.json" 2>&1

    kubectl exec -n kube-system sglang-glm52-2tp8-sglang-1 -- curl -s -X POST "http://localhost:30000/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":500,\"temperature\":0}" \
        > "$TMPDIR/pod1_$pnum.json" 2>&1

    python3 -c "
import json

files = {
    'gateway': '$TMPDIR/gw_$pnum.json',
    'pod-0':   '$TMPDIR/pod0_$pnum.json',
    'pod-1':   '$TMPDIR/pod1_$pnum.json',
}

results = {}
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
        ct = usage.get('completion_tokens', 0)
        rt = usage.get('reasoning_tokens', 0)
        results[name] = content
        print(f'  [{name}] finish={finish} content_len={len(content)} completion_tok={ct} reasoning_tok={rt}')
        if content:
            preview = content.replace(chr(10), ' ')[:120]
            print(f'           content: {preview}')
        else:
            rpreview = reasoning.replace(chr(10), ' ')[:80]
            print(f'           reasoning: {rpreview}...')
    except Exception as e:
        print(f'  [{name}] PARSE ERROR: {e}')

# Check alignment
contents = {k: v for k, v in results.items() if v}
if len(contents) == 3:
    vals = list(contents.values())
    if vals[0] == vals[1] == vals[2]:
        print(f'  ALIGNMENT: IDENTICAL across all 3 endpoints')
    else:
        print(f'  ALIGNMENT: DIFFERENT (semantic equivalence check needed)')
elif len(contents) >= 2:
    vals = list(contents.values())
    if all(v == vals[0] for v in vals):
        print(f'  ALIGNMENT: IDENTICAL among endpoints with content')
    else:
        print(f'  ALIGNMENT: content differs across endpoints')
else:
    print(f'  ALIGNMENT: insufficient content (most hit max_tokens during reasoning)')
"
    echo ""
done

echo "=== Correctness test v2 complete ==="
rm -rf "$TMPDIR"
