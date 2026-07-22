#!/bin/bash
# Concurrency staircase test: 1, 2, 4, 8, 16, 32 concurrent requests
set -uo pipefail

ROUTER="http://21.234.170.159:30001"
API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL="glm-5.2"

PROMPT="Explain what machine learning is in 2 sentences."
MAX_TOKENS=512

echo "=== GLM-5.2 Concurrency Staircase Test (PD + GDR) ==="
echo "Prompt: $PROMPT"
echo "max_tokens: $MAX_TOKENS"
echo ""

for conc in 1 2 4 8 16 32; do
  echo "--- Concurrency: $conc ---"

  # Launch $conc concurrent requests
  pids=()
  results_dir=$(mktemp -d)

  for j in $(seq 1 $conc); do
    (
      start=$(date +%s%3N)
      http_code=$(curl -s -o "$results_dir/resp_$j.json" -w "%{http_code}" \
        -X POST "$ROUTER/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS,\"stream\":false}" \
        --max-time 180)
      end=$(date +%s%3N)
      latency_ms=$((end - start))
      echo "$http_code $latency_ms" > "$results_dir/meta_$j"
    ) &
    pids+=($!)
  done

  # Wait for all
  for pid in "${pids[@]}"; do
    wait $pid
  done

  # Analyze results
  success=0
  fail=0
  total_lat=0
  max_lat=0
  min_lat=999999

  for j in $(seq 1 $conc); do
    meta=$(cat "$results_dir/meta_$j")
    http_code=$(echo "$meta" | awk '{print $1}')
    lat=$(echo "$meta" | awk '{print $2}')

    if [ "$http_code" = "200" ]; then
      success=$((success + 1))
      total_lat=$((total_lat + lat))
      if [ "$lat" -gt "$max_lat" ]; then max_lat=$lat; fi
      if [ "$lat" -lt "$min_lat" ]; then min_lat=$lat; fi
    else
      fail=$((fail + 1))
      echo "  Req $j: HTTP $http_code"
    fi
  done

  if [ "$success" -gt 0 ]; then
    avg_lat=$((total_lat / success))
    echo "  Success: $success/$conc, Fail: $fail"
    echo "  Latency (ms): min=$min_lat avg=$avg_lat max=$max_lat"
  else
    echo "  Success: 0/$conc, Fail: $fail"
  fi
  echo ""

  rm -rf "$results_dir"
  sleep 2
done

echo "=== Staircase Test Complete ==="
