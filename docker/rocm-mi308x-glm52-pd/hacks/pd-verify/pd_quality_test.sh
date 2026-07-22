#!/bin/bash
# Quality test: 10 diverse prompts to verify GLM-5.2 output quality via PD with GDR
set -uo pipefail

ROUTER="http://21.234.170.159:30001"
API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL="glm-5.2"

prompts=(
  "What is 2+2? Answer in one short sentence."
  "Write a haiku about the ocean."
  "Explain what a binary search tree is in 2 sentences."
  "Translate 'Hello, how are you?' to French, Spanish, and Japanese."
  "What is the capital of France? One word answer."
  "Write a Python function that reverses a string. Just the code."
  "Name three primary colors."
  "What is the boiling point of water in Celsius? One sentence."
  "Write a one-sentence story about a cat."
  "What does HTTP stand for? One sentence."
)

echo "=== GLM-5.2 Quality Test (PD + GDR) ==="
echo "Router: $ROUTER"
echo ""

pass=0
fail=0
total=${#prompts[@]}

for i in "${!prompts[@]}"; do
  prompt="${prompts[$i]}"
  echo "--- Prompt $((i+1))/$total ---"
  echo "Q: $prompt"

  start=$(date +%s%3N)
  response=$(curl -s -X POST "$ROUTER/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}],\"max_tokens\":300,\"stream\":false}" \
    --max-time 120 2>&1)
  end=$(date +%s%3N)
  latency_ms=$((end - start))

  content=$(echo "$response" | python3 -c '
import json, sys
try:
    r = json.loads(sys.stdin.read())
    c = r["choices"][0]["message"]["content"]
    fr = r["choices"][0]["finish_reason"]
    usage = r.get("usage", {})
    ct = usage.get("completion_tokens", "?")
    rt = usage.get("reasoning_tokens", "?")
    print("[%s] tokens=%s reasoning=%s" % (fr, ct, rt))
    if c:
        print("A: %s" % c)
    else:
        print("A: (empty content - only reasoning)")
except Exception as e:
    print("ERROR: %s" % e)
' 2>&1)

  echo "$content"
  echo "Latency: ${latency_ms}ms"
  echo ""

  if echo "$response" | grep -q '"finish_reason"'; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
    echo "RAW: $response" | head -5
  fi
done

echo "=== Summary ==="
echo "Pass: $pass / $total"
echo "Fail: $fail / $total"
