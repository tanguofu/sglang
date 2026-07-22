#!/bin/bash
# Benchmark GLM-5.2 2tp8 via gateway HTTPRoute
# Tests 3 scenarios: short (32 in, 256 out), medium (128 in, 256 out), long (2048 in, 256 out)

GATEWAY="https://glm52-2tp8.jmpti.woa.com"
API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
NUM_REQUESTS=32
CONCURRENCY=8

benchmark_scenario() {
    local scenario_name=$1
    local input_len=$2
    local output_len=$3
    local input_text=""

    # Generate input text of approximately input_len tokens (4 chars per token approx)
    local text_len=$((input_len * 4))
    if [ $input_len -le 64 ]; then
        input_text="Hello, please explain machine learning concepts in detail. "
    else
        local repeats=$((text_len / 60 + 1))
        input_text=$(python3 -c "print('Hello, please explain machine learning concepts in detail. ' * $repeats)")
    fi

    echo "============================================================"
    echo "Scenario: $scenario_name (input=$input_len, output=$output_len, reqs=$NUM_REQUESTS, conc=$CONCURRENCY)"
    echo "============================================================"

    local tmpdir=$(mktemp -d)
    local start_time=$(date +%s.%N)

    # Launch concurrent requests
    for i in $(seq 1 $NUM_REQUESTS); do
        (
            local req_start=$(date +%s.%N)
            local response=$(curl -s -w "\n%{time_total}" --max-time 300 \
                "${GATEWAY}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer ${API_KEY}" \
                -d "{
                    \"model\": \"default\",
                    \"messages\": [{\"role\": \"user\", \"content\": \"$input_text\"}],
                    \"max_tokens\": $output_len,
                    \"stream\": false,
                    \"temperature\": 1.0,
                    \"top_p\": 0.95
                }" 2>&1)

            local http_time=$(echo "$response" | tail -1)
            local body=$(echo "$response" | sed '$d')
            local req_end=$(date +%s.%N)
            local req_dur=$(python3 -c "print(f'{$req_end - $req_start:.3f}')")

            # Extract completion tokens and prompt tokens
            local comp_tokens=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")
            local prompt_tokens=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('prompt_tokens',0))" 2>/dev/null || echo "0")
            local finish=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('finish_reason',''))" 2>/dev/null || echo "err")

            echo "$i|$req_dur|$http_time|$comp_tokens|$prompt_tokens|$finish" > "$tmpdir/req_$i.txt"
        ) &

        # Control concurrency
        if [ $(( i % CONCURRENCY )) -eq 0 ]; then
            wait
        fi
    done
    wait

    local end_time=$(date +%s.%N)
    local total_dur=$(python3 -c "print(f'{$end_time - $start_time:.3f}')")

    # Aggregate results
    echo "--- Results ---"
    local total_comp_tokens=0
    local total_prompt_tokens=0
    local total_req_time=0
    local success_count=0
    local fail_count=0
    local max_req_time=0
    local min_req_time=999999

    for f in $tmpdir/req_*.txt; do
        if [ -f "$f" ]; then
            local line=$(cat "$f")
            local idx=$(echo "$line" | cut -d'|' -f1)
            local dur=$(echo "$line" | cut -d'|' -f2)
            local http=$(echo "$line" | cut -d'|' -f3)
            local comp=$(echo "$line" | cut -d'|' -f4)
            local prompt=$(echo "$line" | cut -d'|' -f5)
            local finish=$(echo "$line" | cut -d'|' -f6)

            if [ "$finish" = "length" ] || [ "$finish" = "stop" ]; then
                success_count=$((success_count + 1))
                total_comp_tokens=$((total_comp_tokens + comp))
                total_prompt_tokens=$((total_prompt_tokens + prompt))
                total_req_time=$(python3 -c "print(f'{$total_req_time + $dur:.3f}')")
                max_req_time=$(python3 -c "print(max($max_req_time, $dur))")
                min_req_time=$(python3 -c "print(min($min_req_time, $dur))")
            else
                fail_count=$((fail_count + 1))
                echo "  FAIL req#$idx: finish=$finish dur=${dur}s"
            fi
        fi
    done

    echo "  Success: $success_count / $NUM_REQUESTS, Failed: $fail_count"
    echo "  Total time: ${total_dur}s"
    echo "  Total completion tokens: $total_comp_tokens"
    echo "  Total prompt tokens: $total_prompt_tokens"

    if [ $success_count -gt 0 ]; then
        local throughput=$(python3 -c "print(f'{$total_comp_tokens / $total_dur:.2f}')")
        local avg_req_time=$(python3 -c "print(f'{$total_req_time / $success_count:.3f}')")
        echo "  Throughput: $throughput tok/s (completion)"
        echo "  Avg request time: ${avg_req_time}s"
        echo "  Min/Max request time: ${min_req_time}s / ${max_req_time}s"
    fi

    rm -rf $tmpdir
    echo ""
}

echo "=== GLM-5.2 2tp8 Gateway Benchmark ==="
echo "Gateway: $GATEWAY"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Scenario 1: short input (32 tokens), 256 output
benchmark_scenario "short_c32" 32 256

# Scenario 2: medium input (128 tokens), 256 output
benchmark_scenario "short_c128" 128 256

# Scenario 3: long input (2048 tokens), 256 output
benchmark_scenario "mid_c2048" 2048 256

echo "=== Benchmark Complete ==="
