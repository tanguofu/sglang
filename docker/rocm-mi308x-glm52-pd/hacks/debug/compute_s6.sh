#!/bin/bash
# Compute S6 stats and save as JSON
pd_s6_avg=$(awk '{sum+=$1;n++} END{printf "%.3f", sum/n}' /tmp/bench-results/pd_s6_latency.txt)
pd_s6_total=$(awk '{sum+=$1} END{printf "%.3f", sum}' /tmp/bench-results/pd_s6_latency.txt)
pd_s6_min=$(sort -n /tmp/bench-results/pd_s6_latency.txt | head -1)
pd_s6_max=$(sort -n /tmp/bench-results/pd_s6_latency.txt | tail -1)
pd_s6_p50=$(sort -n /tmp/bench-results/pd_s6_latency.txt | sed -n '4p')
pd_s6_p90=$(sort -n /tmp/bench-results/pd_s6_latency.txt | sed -n '8p')
pd_s6_thr=$(echo "scale=2; 8 / $pd_s6_total" | bc)

tp8_s6_avg=$(awk '{sum+=$1;n++} END{printf "%.3f", sum/n}' /tmp/bench-results/tp8_s6_latency.txt)
tp8_s6_total=$(awk '{sum+=$1} END{printf "%.3f", sum}' /tmp/bench-results/tp8_s6_latency.txt)
tp8_s6_min=$(sort -n /tmp/bench-results/tp8_s6_latency.txt | head -1)
tp8_s6_max=$(sort -n /tmp/bench-results/tp8_s6_latency.txt | tail -1)
tp8_s6_p50=$(sort -n /tmp/bench-results/tp8_s6_latency.txt | sed -n '4p')
tp8_s6_p90=$(sort -n /tmp/bench-results/tp8_s6_latency.txt | sed -n '8p')
tp8_s6_thr=$(echo "scale=2; 8 / $tp8_s6_total" | bc)

jq -n --arg label "pd_s6" --argjson input_tokens 200 --argjson max_tokens 200 --argjson concurrency 4 --argjson num_requests 8 --argjson successful 8 --arg total_time "$pd_s6_total" --arg avg_lat "$pd_s6_avg" --arg min_lat "$pd_s6_min" --arg max_lat "$pd_s6_max" --arg p50_lat "$pd_s6_p50" --arg p90_lat "$pd_s6_p90" --arg p99_lat "$pd_s6_max" --arg throughput "$pd_s6_thr" '{label:$label, input_tokens:$input_tokens, max_tokens:$max_tokens, concurrency:$concurrency, num_requests:$num_requests, successful:$successful, total_time:$total_time, avg_latency:$avg_lat, min_latency:$min_lat, max_latency:$max_lat, p50_latency:$p50_lat, p90_latency:$p90_lat, p99_latency:$p99_lat, throughput_rps:$throughput}' > /tmp/bench-results/pd_s6.json

jq -n --arg label "tp8_s6" --argjson input_tokens 200 --argjson max_tokens 200 --argjson concurrency 4 --argjson num_requests 8 --argjson successful 8 --arg total_time "$tp8_s6_total" --arg avg_lat "$tp8_s6_avg" --arg min_lat "$tp8_s6_min" --arg max_lat "$tp8_s6_max" --arg p50_lat "$tp8_s6_p50" --arg p90_lat "$tp8_s6_p90" --arg p99_lat "$tp8_s6_max" --arg throughput "$tp8_s6_thr" '{label:$label, input_tokens:$input_tokens, max_tokens:$max_tokens, concurrency:$concurrency, num_requests:$num_requests, successful:$successful, total_time:$total_time, avg_latency:$avg_lat, min_latency:$min_lat, max_latency:$max_lat, p50_latency:$p50_lat, p90_latency:$p90_lat, p99_latency:$p99_lat, throughput_rps:$throughput}' > /tmp/bench-results/tp8_s6.json

echo "S6 JSON saved. PD avg=$pd_s6_avg, TP8 avg=$tp8_s6_avg"
