#!/bin/bash
# Benchmark: PD disaggregation vs 2x TP=8 (cache_aware)
# Tests: latency (single request) and throughput (concurrent requests)
# Scenarios: short input/short output, long input/short output, short input/long output

set -euo pipefail

API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
PD_URL="http://glm52-pd.jmpti.woa.com"
TP8_URL="http://glm52-2tp8.jmpti.woa.com"
MODEL="default"
OUTDIR="/tmp/bench-results"
mkdir -p "$OUTDIR"

run_bench() {
    local label="$1"
    local url="$2"
    local input_tokens="$3"
    local max_tokens="$4"
    local concurrency="$5"
    local num_requests="$6"

    # Build prompt of approximate token length
    local prompt=""
    if [ "$input_tokens" -le 50 ]; then
        prompt="Say hello and explain AI briefly."
    elif [ "$input_tokens" -le 500 ]; then
        prompt="Please summarize the key concepts of machine learning, deep learning, and neural networks. Explain supervised learning, unsupervised learning, and reinforcement learning. Describe how gradient descent works, what backpropagation is, and why activation functions matter. Cover convolutional neural networks for images, recurrent neural networks for sequences, and transformer architectures for attention-based models. Discuss overfitting, regularization, dropout, batch normalization, and data augmentation techniques."
    else
        prompt="Write a comprehensive essay about the history of computing, from the earliest mechanical calculators through modern quantum computing. Cover the abacus, the Analytical Engine designed by Charles Babbage, the work of Ada Lovelace, Herman Hollerith's tabulating machines, the vacuum tube era with ENIAC and UNIVAC, the transistor revolution, the development of integrated circuits, the microprocessor, personal computers, the internet, mobile computing, cloud computing, artificial intelligence, machine learning, deep learning, large language models, and quantum computing. Discuss the social and economic impacts of each major advance. Explain the technical principles behind each generation of computing technology. Analyze the contributions of key figures like Alan Turing, John von Neumann, Claude Shannon, Grace Hopper, Gordon Moore, Steve Jobs, Bill Gates, and others. Consider the future trajectory of computing technology and its potential implications for humanity. Include discussion of Moore's Law, Amdahl's Law, and other fundamental principles. Describe the evolution of programming languages from machine code through assembly, FORTRAN, COBOL, C, C++, Java, Python, and modern frameworks. Explain operating systems from batch processing through time-sharing, Unix, Linux, Windows, macOS, iOS, and Android. Cover databases, networking, security, cryptography, and distributed systems. Discuss the rise of open source software and its impact on the industry. Analyze the business models of major technology companies. Consider ethical implications of computing including privacy, surveillance, algorithmic bias, and the digital divide. Speculate on future developments including neuromorphic computing, optical computing, biological computing, and artificial general intelligence."
    fi

    local payload=$(jq -n --arg m "$MODEL" --arg p "$prompt" --argjson mt $max_tokens \
        '{model:$m, max_tokens:$mt, messages:[{role:"user", content:$p}]}')

    local result_file="$OUTDIR/${label}.json"
    local latency_file="$OUTDIR/${label}_latency.txt"

    echo "[$(date +%H:%M:%S)] Running $label: input~${input_tokens}tok, max_out=${max_tokens}, concurrency=${concurrency}, num=${num_requests}"

    # Warmup request
    curl -s --max-time 120 -X POST "$url/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "$payload" > /dev/null 2>&1 || true

    # Benchmark run
    local start_ts=$(date +%s.%N)

    if [ "$concurrency" -eq 1 ]; then
        # Sequential
        for i in $(seq 1 $num_requests); do
            local t1=$(date +%s.%N)
            local resp=$(curl -s -w "\n%{http_code} %{time_total}" --max-time 120 \
                -X POST "$url/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $API_KEY" \
                -d "$payload" 2>&1)
            local http_code=$(echo "$resp" | tail -1 | awk '{print $1}')
            local req_time=$(echo "$resp" | tail -1 | awk '{print $2}')
            echo "$req_time" >> "$latency_file"
        done
    else
        # Concurrent using xargs
        seq 1 $num_requests | xargs -P $concurrency -I {} bash -c "
            curl -s -w '%{time_total}\n' -o /dev/null --max-time 120 \
                -X POST '$url/v1/chat/completions' \
                -H 'Content-Type: application/json' \
                -H 'Authorization: Bearer $API_KEY' \
                -d '$payload' 2>&1
        " >> "$latency_file"
    fi

    local end_ts=$(date +%s.%N)
    local total_time=$(echo "$end_ts - $start_ts" | bc)
    local successful=$(wc -l < "$latency_file")

    # Compute stats
    local avg_lat=$(awk '{sum+=$1; n++} END {if(n>0) printf "%.3f", sum/n}' "$latency_file")
    local min_lat=$(awk 'NR==1 || $1<min {min=$1} END {printf "%.3f", min}' "$latency_file")
    local max_lat=$(awk 'NR==1 || $1>max {max=$1} END {printf "%.3f", max}' "$latency_file")
    local p50_lat=$(sort -n "$latency_file" | awk 'NR==int(NR/2)+1 {printf "%.3f", $1}')
    local p90_lat=$(sort -n "$latency_file" | awk 'NR==int(NR*0.9)+1 {printf "%.3f", $1}')
    local p99_lat=$(sort -n "$latency_file" | awk 'NR==int(NR*0.99)+1 {printf "%.3f", $1}')
    local throughput=$(echo "scale=2; $successful / $total_time" | bc)

    jq -n \
        --arg label "$label" \
        --argjson input_tokens "$input_tokens" \
        --argjson max_tokens "$max_tokens" \
        --argjson concurrency "$concurrency" \
        --argjson num_requests "$num_requests" \
        --argjson successful "$successful" \
        --arg total_time "$total_time" \
        --arg avg_lat "$avg_lat" \
        --arg min_lat "$min_lat" \
        --arg max_lat "$max_lat" \
        --arg p50_lat "$p50_lat" \
        --arg p90_lat "$p90_lat" \
        --arg p99_lat "$p99_lat" \
        --arg throughput "$throughput" \
        '{label:$label, input_tokens:$input_tokens, max_tokens:$max_tokens, concurrency:$concurrency, num_requests:$num_requests, successful:$successful, total_time:$total_time, avg_latency:$avg_lat, min_latency:$min_lat, max_latency:$max_lat, p50_latency:$p50_lat, p90_latency:$p90_lat, p99_latency:$p99_lat, throughput_rps:$throughput}' \
        > "$result_file"

    echo "  -> $successful/$num_requests ok, total=${total_time}s, avg=${avg_lat}s, p50=${p50_lat}s, p90=${p90_lat}s, throughput=${throughput} rps"
    rm -f "$latency_file"
}

echo "============================================================"
echo "  SGLang Benchmark: PD Disaggregation vs 2x TP=8"
echo "  PD URL:   $PD_URL"
echo "  TP8 URL:  $TP8_URL"
echo "  Time:     $(date)"
echo "============================================================"

# Scenario 1: Short input, short output (latency-sensitive)
echo ""
echo "=== Scenario 1: Short input (~30 tok), short output (50 tok), sequential x10 ==="
run_bench "pd_s1_short"  "$PD_URL"  30  50 1 10
run_bench "tp8_s1_short" "$TP8_URL" 30  50 1 10

# Scenario 2: Short input, long output (decode-heavy)
echo ""
echo "=== Scenario 2: Short input (~30 tok), long output (500 tok), sequential x5 ==="
run_bench "pd_s2_decode"  "$PD_URL"  30  500 1 5
run_bench "tp8_s2_decode" "$TP8_URL" 30  500 1 5

# Scenario 3: Long input (~500 tok), short output (50 tok) (prefill-heavy)
echo ""
echo "=== Scenario 3: Long input (~500 tok), short output (50 tok), sequential x5 ==="
run_bench "pd_s3_prefill"  "$PD_URL"  500  50 1 5
run_bench "tp8_s3_prefill" "$TP8_URL" 500  50 1 5

# Scenario 4: Short input, medium output, concurrency=4
echo ""
echo "=== Scenario 4: Short input (~30 tok), medium output (200 tok), concurrency=4, x16 ==="
run_bench "pd_s4_conc4"  "$PD_URL"  30  200 4 16
run_bench "tp8_s4_conc4" "$TP8_URL" 30  200 4 16

# Scenario 5: Short input, medium output, concurrency=8
echo ""
echo "=== Scenario 5: Short input (~30 tok), medium output (200 tok), concurrency=8, x32 ==="
run_bench "pd_s5_conc8"  "$PD_URL"  30  200 8 32
run_bench "tp8_s5_conc8" "$TP8_URL" 30  200 8 32

# Scenario 6: Long input (~2000 tok), medium output (200 tok), concurrency=4
echo ""
echo "=== Scenario 6: Long input (~2000 tok), medium output (200 tok), concurrency=4, x8 ==="
run_bench "pd_s6_long"  "$PD_URL"  2000  200 4 8
run_bench "tp8_s6_long" "$TP8_URL" 2000  200 4 8

echo ""
echo "============================================================"
echo "  Benchmark Complete: $(date)"
echo "============================================================"
echo ""
echo "=== Results Summary ==="
for f in "$OUTDIR"/*.json; do
    jq -r '[.label, .input_tokens, .max_tokens, .concurrency, .num_requests, .successful, .total_time, .avg_latency, .p50_latency, .p90_latency, .p99_latency, .throughput_rps] | @tsv' "$f"
done | column -t -N "label,input_tok,max_tok,conc,num,ok,total_s,avg_s,p50_s,p90_s,p99_s,thr_rps"
