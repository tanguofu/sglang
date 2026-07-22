#!/bin/bash
# Benchmark GLM-5.2 2tp8 via gateway — capture TTFT and TPOT via streaming
GATEWAY="https://glm52-2tp8.jmpti.woa.com"
API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
NUM_REQUESTS=32
CONCURRENCY=8

benchmark_scenario() {
    local scenario_name=$1
    local input_len=$2
    local output_len=$3

    # Generate input text ~input_len tokens (4 chars/token)
    local text_len=$((input_len * 4))
    local input_text
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

    for i in $(seq 1 $NUM_REQUESTS); do
        (
            local req_start=$(date +%s.%N)
            # Use streaming to capture TTFT (time to first token)
            local stream_file="$tmpdir/stream_$i.txt"
            curl -sN --max-time 300 \
                "${GATEWAY}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer ${API_KEY}" \
                -d "{
                    \"model\": \"default\",
                    \"messages\": [{\"role\": \"user\", \"content\": \"$input_text\"}],
                    \"max_tokens\": $output_len,
                    \"stream\": true,
                    \"temperature\": 1.0,
                    \"top_p\": 0.95,
                    \"stream_options\": {\"include_usage\": true}
                }" > "$stream_file" 2>&1

            local req_end=$(date +%s.%N)
            local req_dur=$(python3 -c "print(f'{$req_end - $req_start:.3f}')")

            # Parse SSE stream: first content chunk = TTFT, last = total time
            python3 << PYEOF >> "$tmpdir/result_$i.txt"
import json, sys
ttft = None
first_chunk_time = None
content_chunks = 0
total_tokens = 0
prompt_tokens = 0
finish = ""
req_start = $req_start
req_dur = $req_dur

with open("$stream_file") as f:
    for line in f:
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except:
            continue
        # First chunk with content = TTFT
        if "choices" in obj and obj["choices"]:
            ch = obj["choices"][0]
            delta = ch.get("delta", {})
            if "content" in delta and delta["content"] and first_chunk_time is None:
                first_chunk_time = req_start  # approximation: first content seen
                ttft = req_dur  # placeholder, will recalc
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
        if "usage" in obj and obj["usage"]:
            total_tokens = obj["usage"].get("completion_tokens", 0)
            prompt_tokens = obj["usage"].get("prompt_tokens", 0)

# We need TTFT: time from req_start to first content chunk
# Re-parse with timestamps — use file mtime as approximation isn't accurate
# Instead, use curl's --write-out with time_starttransfer
print(f"{total_tokens}|{prompt_tokens}|{finish}|{req_dur}")
PYEOF
        ) &

        if [ $(( i % CONCURRENCY )) -eq 0 ]; then
            wait
        fi
    done
    wait

    local end_time=$(date +%s.%N)
    local total_dur=$(python3 -c "print(f'{$end_time - $start_time:.3f}')")

    # Better approach: use curl --write-out for TTFT
    # Re-run with proper timing capture
    echo "  Re-running with TTFT capture..."
    local ttft_dir=$(mktemp -d)

    for i in $(seq 1 $NUM_REQUESTS); do
        (
            local out_file="$ttft_dir/req_$i.json"
            local timing_file="$ttft_dir/timing_$i.txt"

            curl -sN --max-time 300 \
                -o "$out_file" \
                -w "time_total=%{time_total}\ntime_starttransfer=%{time_starttransfer}\nsize_download=%{size_download}\n" \
                "${GATEWAY}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer ${API_KEY}" \
                -d "{
                    \"model\": \"default\",
                    \"messages\": [{\"role\": \"user\", \"content\": \"$input_text\"}],
                    \"max_tokens\": $output_len,
                    \"stream\": true,
                    \"temperature\": 1.0,
                    \"top_p\": 0.95,
                    \"stream_options\": {\"include_usage\": true}
                }" > "$timing_file" 2>&1
        ) &

        if [ $(( i % CONCURRENCY )) -eq 0 ]; then
            wait
        fi
    done
    wait

    # Aggregate
    echo "--- Results ---"
    local total_comp_tokens=0
    local total_ttft=0
    local total_tpot=0
    local total_req_time=0
    local success_count=0
    local fail_count=0
    local max_ttft=0
    local min_ttft=999999
    local max_tpot=0
    local min_tpot=999999

    for f in $ttft_dir/timing_*.txt; do
        if [ -f "$f" ]; then
            local idx=$(basename "$f" | sed 's/timing_//;s/.txt//')
            local time_total=$(grep "time_total=" "$f" | cut -d'=' -f2)
            local time_starttransfer=$(grep "time_starttransfer=" "$f" | cut -d'=' -f2)

            # Count tokens from stream file
            local comp_tokens=$(python3 -c "
import json
count = 0
with open('$ttft_dir/req_$idx.json') as f:
    for line in f:
        line = line.strip()
        if not line.startswith('data: '):
            continue
        data = line[6:]
        if data == '[DONE]':
            break
        try:
            obj = json.loads(data)
            if 'usage' in obj and obj['usage']:
                count = obj['usage'].get('completion_tokens', 0)
        except:
            pass
print(count)
" 2>/dev/null || echo "0")

            if [ "$comp_tokens" -gt 0 ] && [ -n "$time_total" ] && [ -n "$time_starttransfer" ]; then
                success_count=$((success_count + 1))
                total_comp_tokens=$((total_comp_tokens + comp_tokens))
                total_ttft=$(python3 -c "print(f'{$total_ttft + $time_starttransfer:.6f}')")
                # TPOT = (total - ttft) / (output_tokens - 1)
                local tpot=$(python3 -c "
ttot = float('$time_total')
ttft = float('$time_starttransfer')
ct = int('$comp_tokens')
if ct > 1:
    print(f'{(ttot - ttft) / (ct - 1):.6f}')
else:
    print('0')
")
                total_tpot=$(python3 -c "print(f'{$total_tpot + $tpot:.6f}')")
                total_req_time=$(python3 -c "print(f'{$total_req_time + $time_total:.6f}')")
                max_ttft=$(python3 -c "print(max($max_ttft, $time_starttransfer))")
                min_ttft=$(python3 -c "print(min($min_ttft, $time_starttransfer))")
                max_tpot=$(python3 -c "print(max($max_tpot, $tpot))")
                min_tpot=$(python3 -c "print(min($min_tpot, $tpot))")
            else
                fail_count=$((fail_count + 1))
            fi
        fi
    done

    echo "  Success: $success_count / $NUM_REQUESTS, Failed: $fail_count"
    echo "  Total time: ${total_dur}s"
    echo "  Total completion tokens: $total_comp_tokens"

    if [ $success_count -gt 0 ]; then
        local throughput=$(python3 -c "print(f'{$total_comp_tokens / $total_dur:.2f}')")
        local avg_ttft=$(python3 -c "print(f'{$total_ttft / $success_count * 1000:.1f}')")
        local avg_tpot=$(python3 -c "print(f'{$total_tpot / $success_count * 1000:.1f}')")
        local min_ttft_ms=$(python3 -c "print(f'{$min_ttft * 1000:.1f}')")
        local max_ttft_ms=$(python3 -c "print(f'{$max_ttft * 1000:.1f}')")
        local min_tpot_ms=$(python3 -c "print(f'{$min_tpot * 1000:.1f}')")
        local max_tpot_ms=$(python3 -c "print(f'{$max_tpot * 1000:.1f}')")
        echo "  Throughput: $throughput tok/s (completion)"
        echo "  TTFT (avg): ${avg_ttft} ms | min/max: ${min_ttft_ms} / ${max_ttft_ms} ms"
        echo "  TPOT (avg): ${avg_tpot} ms | min/max: ${min_tpot_ms} / ${max_tpot_ms} ms"
    fi

    rm -rf $tmpdir $ttft_dir
    echo ""
}

echo "=== GLM-5.2 2tp8 Gateway Benchmark (TTFT/TPOT) ==="
echo "Gateway: $GATEWAY"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

benchmark_scenario "short_c32" 32 256
benchmark_scenario "short_c128" 128 256
benchmark_scenario "mid_c2048" 2048 256

echo "=== Benchmark Complete ==="
