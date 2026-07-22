#!/bin/bash
# Gateway benchmark — 3 scenarios, non-streaming
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
    local start=$(date +%s.%N)
    local tmpdir=$(mktemp -d)

    for i in $(seq 1 $total); do
        (
            curl -s -o "$tmpdir/resp_$i.json" -w "%{http_code} %{time_total}\n" \
                -X POST "$API_URL" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $API_KEY" \
                -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$max_tokens}" \
                > "$tmpdir/metrics_$i.txt" 2>&1
        ) &
        if (( i % concurrency == 0 )); then wait; fi
    done
    wait

    local end=$(date +%s.%N)
    local elapsed=$(echo "$end - $start" | bc)

    local success=0
    local fail=0
    local total_input=0
    local total_output=0
    local total_time=0

    for i in $(seq 1 $total); do
        code=$(cat "$tmpdir/metrics_$i.txt" | awk '{print $1}')
        if [ "$code" = "200" ]; then
            success=$((success + 1))
            ct=$(python3 -c "import json; r=json.load(open('$tmpdir/resp_$i.json')); print(r.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
            total_output=$((total_output + ct))
        else
            fail=$((fail + 1))
        fi
    done

    local throughput=$(echo "scale=2; $total_output / $elapsed" | bc 2>/dev/null)
    echo "  Success: $success/$total (Fail: $fail)"
    echo "  Total output tokens: $total_output"
    echo "  Elapsed: ${elapsed}s"
    echo "  Throughput: ${throughput} tok/s"
    echo ""

    rm -rf "$tmpdir"
}

# Scenario 1: short_c32 — 32 input, 256 output, 8 concurrency
run_scenario "short_c32" "Describe the key features of AMD MI300 series GPUs." 256 8 32

# Scenario 2: short_c128 — 128 input, 512 output, 8 concurrency
run_scenario "short_c128" "Explain the architecture of transformer models, including attention mechanisms, feed-forward networks, and layer normalization. Also describe how positional encoding works." 512 8 32

# Scenario 3: mid_c2048 — 2048 input, 512 output, 4 concurrency
LONG_PROMPT="You are a senior software architect. I need a detailed design document for a distributed inference system. The system should support: 1) Multiple model serving backends (vLLM, SGLang, TGI) behind a unified API. 2) Request routing based on prefix cache hit rates. 3) Automatic scaling based on queue depth. 4) Multi-tenant isolation with per-tenant rate limits. 5) Observability with Prometheus metrics. 6) A/B testing framework for model comparison. 7) Fallback chains for high availability. 8) Cost optimization via spot instance support. 9) Model versioning and canary deployments. 10) Request batching for throughput optimization. For each component, describe the interface, implementation strategy, and failure modes. Include sequence diagrams for the critical paths."
run_scenario "mid_c2048" "$LONG_PROMPT" 512 4 16

echo "=== Benchmark complete ==="
