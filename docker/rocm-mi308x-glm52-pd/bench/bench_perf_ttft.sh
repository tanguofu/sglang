#!/bin/bash
# Full performance benchmark — 3 scenarios with TTFT/TPOT via streaming
# Uses curl --write-out time_starttransfer for TTFT, calculates TPOT from total time.
set -uo pipefail

API_URL="http://glm52-2tp8.jmpti.woa.com/v1/chat/completions"
API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL="glm-5.2"

run_scenario() {
    local name=$1
    local prompt=$2
    local max_tokens=$3
    local concurrency=$4
    local total=$5

    echo "=== $name (concurrency=$concurrency, total=$total, max_tokens=$max_tokens) ==="
    local tmpdir=$(mktemp -d)

    # Start time
    local start_ts=$(date +%s.%N)

    for i in $(seq 1 $total); do
        (
            # Streaming request to get TTFT
            # time_starttransfer = TTFT (time to first byte)
            # time_total = total request time
            curl -s -o "$tmpdir/resp_$i.json" \
                -w "%{http_code} %{time_starttransfer} %{time_total}\n" \
                -X POST "$API_URL" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $API_KEY" \
                -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$max_tokens,\"stream\":true}" \
                > "$tmpdir/metrics_$i.txt" 2>&1
        ) &
        if (( i % concurrency == 0 )); then wait; fi
    done
    wait

    local end_ts=$(date +%s.%N)
    local wall_elapsed=$(echo "$end_ts - $start_ts" | bc)

    # Aggregate metrics
    python3 -c "
import json, os, re

tmpdir = '$tmpdir'
total = $total
max_tokens = $max_tokens

success = 0
fail = 0
total_output = 0
total_input = 0
ttft_list = []
total_time_list = []
tpot_list = []

for i in range(1, total + 1):
    metrics_path = os.path.join(tmpdir, f'metrics_{i}.txt')
    resp_path = os.path.join(tmpdir, f'resp_{i}.json')

    try:
        with open(metrics_path) as f:
            metric_line = f.read().strip()
        parts = metric_line.split()
        code = parts[0]
        ttft = float(parts[1]) if len(parts) > 1 else 0
        total_time = float(parts[2]) if len(parts) > 2 else 0
    except:
        code = '000'
        ttft = 0
        total_time = 0

    if code != '200':
        fail += 1
        continue

    success += 1
    ttft_list.append(ttft)
    total_time_list.append(total_time)

    # Parse streaming response to count output tokens
    # Streaming response is SSE: data: {json}\n\n
    output_tokens = 0
    try:
        with open(resp_path) as f:
            content = f.read()
        for match in re.finditer(r'data: (.+)', content):
            chunk_str = match.group(1).strip()
            if chunk_str == '[DONE]':
                continue
            try:
                chunk = json.loads(chunk_str)
                if chunk.get('usage') and chunk['usage'].get('completion_tokens'):
                    output_tokens = chunk['usage']['completion_tokens']
            except:
                pass
    except:
        pass

    if output_tokens == 0:
        # Fallback: count content characters / 4 (rough estimate)
        try:
            with open(resp_path) as f:
                content = f.read()
            for match in re.finditer(r'data: (.+)', content):
                chunk_str = match.group(1).strip()
                if chunk_str == '[DONE]':
                    continue
                try:
                    chunk = json.loads(chunk_str)
                    delta = chunk.get('choices', [{}])[0].get('delta', {})
                    if delta.get('content'):
                        output_tokens += len(delta['content']) // 4
                    if delta.get('reasoning_content'):
                        output_tokens += len(delta['reasoning_content']) // 4
                except:
                    pass
        except:
            pass

    total_output += output_tokens

    # TPOT = (total_time - ttft) / output_tokens
    if output_tokens > 0 and total_time > ttft:
        tpot = (total_time - ttft) / output_tokens * 1000  # ms
        tpot_list.append(tpot)

wall_elapsed = $wall_elapsed

print(f'  Success: {success}/{total} (Fail: {fail})')
print(f'  Total output tokens: {total_output}')
print(f'  Wall elapsed: {wall_elapsed:.2f}s')

if total_output > 0:
    throughput = total_output / wall_elapsed
    print(f'  Throughput: {throughput:.2f} tok/s')

if ttft_list:
    avg_ttft = sum(ttft_list) / len(ttft_list)
    min_ttft = min(ttft_list)
    max_ttft = max(ttft_list)
    p50_ttft = sorted(ttft_list)[len(ttft_list)//2]
    print(f'  TTFT (ms): avg={avg_ttft*1000:.0f} min={min_ttft*1000:.0f} p50={p50_ttft*1000:.0f} max={max_ttft*1000:.0f}')

if tpot_list:
    avg_tpot = sum(tpot_list) / len(tpot_list)
    min_tpot = min(tpot_list)
    max_tpot = max(tpot_list)
    p50_tpot = sorted(tpot_list)[len(tpot_list)//2]
    print(f'  TPOT (ms): avg={avg_tpot:.1f} min={min_tpot:.1f} p50={p50_tpot:.1f} max={max_tpot:.1f}')

print()
"

    rm -rf "$tmpdir"
}

# Scenario 1: short_c32 — 32 input tokens, 256 output, 8 concurrency
run_scenario "short_c32" "Describe the key features of AMD MI300 series GPUs." 256 8 32

# Scenario 2: short_c128 — 128 input tokens, 512 output, 8 concurrency
run_scenario "short_c128" "Explain the architecture of transformer models, including attention mechanisms, feed-forward networks, and layer normalization. Also describe how positional encoding works." 512 8 32

# Scenario 3: mid_c2048 — 2048 input tokens, 512 output, 4 concurrency
LONG_PROMPT="You are a senior software architect. I need a detailed design document for a distributed inference system. The system should support: 1) Multiple model serving backends (vLLM, SGLang, TGI) behind a unified API. 2) Request routing based on prefix cache hit rates. 3) Automatic scaling based on queue depth. 4) Multi-tenant isolation with per-tenant rate limits. 5) Observability with Prometheus metrics. 6) A/B testing framework for model comparison. 7) Fallback chains for high availability. 8) Cost optimization via spot instance support. 9) Model versioning and canary deployments. 10) Request batching for throughput optimization. For each component, describe the interface, implementation strategy, and failure modes. Include sequence diagrams for the critical paths."
run_scenario "mid_c2048" "$LONG_PROMPT" 512 4 16

echo "=== Benchmark complete ==="
