#!/bin/bash
# Concurrent stress test to trigger 503 + capture metrics before/after
echo "=== Pre-test metrics snapshot ==="
kubectl exec -n kube-system sglang-glm52-2tp8-router-697df7c955-2zrcc -- curl -s http://127.0.0.1:29000/metrics -m 10 2>&1 | grep -E "^smg_worker_cb_outcomes_total|^smg_worker_cb_state|^smg_worker_health|^smg_worker_requests_active|^smg_worker_cb_consecutive|^smg_http_responses_total|^smg_router_upstream_responses_total|^smg_router_request_errors_total" | sort

echo ""
echo "=== Concurrent stress test: 20 parallel requests x 3 waves ==="
total_ok=0
total_fail=0
total_503=0
for wave in 1 2 3; do
  echo "--- Wave $wave ---"
  pids=""
  results_dir=$(mktemp -d)
  for i in $(seq 1 20); do
    (
      code=$(curl -s -o /dev/null -w "%{http_code}" https://glm52-2tp8.jmpti.woa.com/v1/chat/completions \
        -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
        -H "Content-Type: application/json" \
        -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is 2+2? Just the number."}],"max_tokens":1500,"stream":false}' \
        -m 60)
      echo "$code" > "$results_dir/req_$i.txt"
    ) &
    pids="$pids $!"
  done
  wait
  
  wave_ok=0
  wave_fail=0
  wave_503=0
  for f in $results_dir/*.txt; do
    code=$(cat "$f")
    case "$code" in
      200) wave_ok=$((wave_ok + 1)) ;;
      503) wave_503=$((wave_503 + 1)) ;;
      *) wave_fail=$((wave_fail + 1)) ;;
    esac
  done
  rm -rf "$results_dir"
  echo "  Wave $wave: $wave_ok OK, $wave_503 503, $wave_fail other"
  total_ok=$((total_ok + wave_ok))
  total_fail=$((total_fail + wave_fail))
  total_503=$((total_503 + wave_503))
done

echo ""
echo "=== Stress test summary ==="
echo "Total: $total_ok OK, $total_503 503, $total_fail other (out of $((20 * 3)) requests)"

echo ""
echo "=== Post-test metrics snapshot ==="
kubectl exec -n kube-system sglang-glm52-2tp8-router-697df7c955-2zrcc -- curl -s http://127.0.0.1:29000/metrics -m 10 2>&1 | grep -E "^smg_worker_cb_outcomes_total|^smg_worker_cb_state|^smg_worker_health|^smg_worker_requests_active|^smg_worker_cb_consecutive|^smg_http_responses_total|^smg_router_upstream_responses_total|^smg_router_request_errors_total" | sort
