#!/bin/bash
# Benchmark: PD disaggregation (1p1d) vs 2x TP=8 (cache_aware router)
# Runs INSIDE the cluster via kubectl exec for fair comparison (bypass HTTPRoute differences)
# Both go through their respective routers.

set -euo pipefail

API_KEY="sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
# PD router: sglang-1p1d-router:30011 (model: default)
# TP8 router: sglang-glm52-2tp8-router:30001 (model: glm-5.2)
PD_URL="http://sglang-1p1d-router.kube-system.svc.cluster.local:30011"
TP8_URL="http://sglang-glm52-2tp8-router.kube-system.svc.cluster.local:30001"
PD_MODEL="default"
TP8_MODEL="glm-5.2"
OUTDIR="/tmp/bench-results"
mkdir -p "$OUTDIR"

# Runner pod (use prefill pod as the curl client)
CLIENT_POD="sglang-1p1d-prefill-0"

run_bench() {
    local label="$1"
    local url="$2"
    local model="$3"
    local input_tokens="$4"
    local max_tokens="$5"
    local concurrency="$6"
    local num_requests="$7"

    # Build prompt of approximate token length
    local prompt
    if [ "$input_tokens" -le 50 ]; then
        prompt="Say hello and explain AI briefly."
    elif [ "$input_tokens" -le 500 ]; then
        prompt="Please summarize the key concepts of machine learning, deep learning, and neural networks. Explain supervised learning, unsupervised learning, and reinforcement learning. Describe how gradient descent works, what backpropagation is, and why activation functions matter. Cover convolutional neural networks for images, recurrent neural networks for sequences, and transformer architectures for attention-based models. Discuss overfitting, regularization, dropout, batch normalization, and data augmentation techniques."
    else
        prompt="Write a comprehensive essay about the history of computing, from the earliest mechanical calculators through modern quantum computing. Cover the abacus, the Analytical Engine designed by Charles Babbage, the work of Ada Lovelace, Herman Hollerith's tabulating machines, the vacuum tube era with ENIAC and UNIVAC, the transistor revolution, the development of integrated circuits, the microprocessor, personal computers, the internet, mobile computing, cloud computing, artificial intelligence, machine learning, deep learning, large language models, and quantum computing. Discuss the social and economic impacts of each major advance. Explain the technical principles behind each generation of computing technology. Analyze the contributions of key figures like Alan Turing, John von Neumann, Claude Shannon, Grace Hopper, Gordon Moore, Steve Jobs, Bill Gates, and others. Consider the future trajectory of computing technology and its potential implications for humanity. Include discussion of Moore's Law, Amdahl's Law, and other fundamental principles. Describe the evolution of programming languages from machine code through assembly, FORTRAN, COBOL, C, C++, Java, Python, and modern frameworks. Explain operating systems from batch processing through time-sharing, Unix, Linux, Windows, macOS, iOS, and Android. Cover databases, networking, security, cryptography, and distributed systems. Discuss the rise of open source software and its impact on the industry. Analyze the business models of major technology companies. Consider ethical implications of computing including privacy, surveillance, algorithmic bias, and the digital divide. Speculate on future developments including neuromorphic computing, optical computing, biological computing, and artificial general intelligence."
    fi

    local payload
    payload=$(jq -n --arg m "$model" --arg p "$prompt" --argjson mt "$max_tokens" \
        '{model:$m, max_tokens:$mt, messages:[{role:"user", content:$p}]}')

    local result_file="$OUTDIR/${label}.json"
    local latency_file="$OUTDIR/${label}_latency.txt"
    rm -f "$latency_file"

    echo "[$(date +%H:%M:%S)] $label: in~${input_tokens}tok, out=${max_tokens}, conc=${concurrency}, n=${num_requests}"

    # Warmup
    kubectl exec -n kube-system "$CLIENT_POD" -- curl -s --max-time 120 -o /dev/null \
        -X POST "$url/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "$payload" 2>&1 || true

    # Benchmark
    local start_ts=$(date +%s.%N)

    if [ "$concurrency" -eq 1 ]; then
        for i in $(seq 1 "$num_requests"); do
            kubectl exec -n kube-system "$CLIENT_POD" -- curl -s -w "%{time_total}\n" -o /dev/null --max-time 180 \
                -X POST "$url/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $API_KEY" \
                -d "$payload" 2>&1 >> "$latency_file"
        done
    else
        # Use kubectl exec with a script that runs concurrent curls
        kubectl exec -n kube-system "$CLIENT_POD" -- bash -c "
            seq 1 $num_requests | xargs -P $concurrency -I {} curl -s -w '%{time_total}\n' -o /dev/null --max-time 180 \
                -X POST '$url/v1/chat/completions' \
                -H 'Content-Type: application/json' \
                -H 'Authorization: Bearer $API_KEY' \
                -d '$payload' 2>&1
        " >> "$latency_file"
    fi

    local end_ts=$(date +%s.%N)
    local total_time=$(echo "$end_ts - $start_ts" | bc)
    local successful=$(wc -l < "$latency_file" | tr -d ' ')

    # Stats
    local avg_lat=$(awk '{sum+=$1; n++} END {if(n>0) printf "%.3f", sum/n}' "$latency_file")
    local min_lat=$(sort -n "$latency_file" | head -1 | awk '{printf "%.3f", $1}')
    local max_lat=$(sort -n "$latency_file" | tail -1 | awk '{printf "%.3f", $1}')
    local sorted_file="${latency_file}.sorted"
    sort -n "$latency_file" > "$sorted_file"
    local n_lines=$(wc -l < "$sorted_file" | tr -d ' ')
    local p50_idx=$(( (n_lines + 1) / 2 ))
    local p90_idx=$(( (n_lines * 9) / 10 + 1 ))
    local p99_idx=$(( (n_lines * 99) / 100 + 1 ))
    [ "$p90_idx" -gt "$n_lines" ] && p90_idx=$n_lines
    [ "$p99_idx" -gt "$n_lines" ] && p99_idx=$n_lines
    local p50_lat=$(sed -n "${p50_idx}p" "$sorted_file" | awk '{printf "%.3f", $1}')
    local p90_lat=$(sed -n "${p90_idx}p" "$sorted_file" | awk '{printf "%.3f", $1}')
    local p99_lat=$(sed -n "${p99_idx}p" "$sorted_file" | awk '{printf "%.3f", $1}')
    rm -f "$sorted_file"
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

    echo "  -> $successful/$num_requests ok, total=${total_time}s, avg=${avg_lat}s, p50=${p50_lat}s, p90=${p90_lat}s, thr=${throughput} rps"
    rm -f "$latency_file"
}

echo "============================================================"
echo "  SGLang Benchmark: PD (1p1d) vs 2x TP=8 (cache_aware)"
echo "  PD router:   $PD_URL (model=$PD_MODEL)"
echo "  TP8 router:  $TP8_URL (model=$TP8_MODEL)"
echo "  Client pod:  $CLIENT_POD"
echo "  Time:        $(date)"
echo "============================================================"

# Scenario 1: Short in, short out — latency test
echo ""
echo "=== S1: Short in (~30tok), short out (50tok), seq x10 ==="
run_bench "pd_s1"  "$PD_URL"  "$PD_MODEL"  30  50  1 10
run_bench "tp8_s1" "$TP8_URL" "$TP8_MODEL" 30  50  1 10

# Scenario 2: Short in, long out — decode-heavy
echo ""
echo "=== S2: Short in (~30tok), long out (500tok), seq x5 ==="
run_bench "pd_s2"  "$PD_URL"  "$PD_MODEL"  30  500 1 5
run_bench "tp8_s2" "$TP8_URL" "$TP8_MODEL" 30  500 1 5

# Scenario 3: Long in (~500tok), short out — prefill-heavy
echo ""
echo "=== S3: Long in (~500tok), short out (50tok), seq x5 ==="
run_bench "pd_s3"  "$PD_URL"  "$PD_MODEL"  500  50  1 5
run_bench "tp8_s3" "$TP8_URL" "$TP8_MODEL" 500  50  1 5

# Scenario 4: Short in, medium out, concurrency=4
echo ""
echo "=== S4: Short in (~30tok), med out (200tok), conc=4, x16 ==="
run_bench "pd_s4"  "$PD_URL"  "$PD_MODEL"  30  200 4 16
run_bench "tp8_s4" "$TP8_URL" "$TP8_MODEL" 30  200 4 16

# Scenario 5: Short in, medium out, concurrency=8
echo ""
echo "=== S5: Short in (~30tok), med out (200tok), conc=8, x32 ==="
run_bench "pd_s5"  "$PD_URL"  "$PD_MODEL"  30  200 8 32
run_bench "tp8_s5" "$TP8_URL" "$TP8_MODEL" 30  200 8 32

# Scenario 6: Long in (~2000tok), medium out, conc=4
echo ""
echo "=== S6: Long in (~2000tok), med out (200tok), conc=4, x8 ==="
run_bench "pd_s6"  "$PD_URL"  "$PD_MODEL"  2000  200 4 8
run_bench "tp8_s6" "$TP8_URL" "$TP8_MODEL" 2000  200 4 8

echo ""
echo "============================================================"
echo "  Benchmark Complete: $(date)"
echo "============================================================"
echo ""
echo "=== Results Summary ==="
printf "%-12s %7s %7s %5s %5s %5s %8s %8s %8s %8s %8s %8s\n" "label" "in_tok" "max_tok" "conc" "num" "ok" "tot_s" "avg_s" "p50_s" "p90_s" "p99_s" "rps"
echo "-----------------------------------------------------------------------------------------------"
for f in "$OUTDIR"/*.json; do
    jq -r '"\(.label) \(.input_tokens) \(.max_tokens) \(.concurrency) \(.num_requests) \(.successful) \(.total_time) \(.avg_latency) \(.p50_latency) \(.p90_latency) \(.p99_latency) \(.throughput_rps)"' "$f"
done | awk '{printf "%-12s %7s %7s %5s %5s %5s %8s %8s %8s %8s %8s %8s\n",$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12}'
