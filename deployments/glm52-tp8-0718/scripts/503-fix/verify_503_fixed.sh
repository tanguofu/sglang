#!/bin/bash
# Verify 503 elimination after service selector fix
echo "=== Pre-test metrics ==="
kubectl exec -n kube-system sglang-glm52-2tp8-router-697df7c955-2zrcc -- curl -s http://127.0.0.1:29000/metrics -m 10 2>&1 | grep -E "^smg_http_responses_total|^smg_router_upstream_responses_total|^smg_worker_cb_outcomes_total|^smg_worker_cb_state" | sort

echo ""
echo "=== Test 1: 20 sequential requests ==="
ok=0; fail=0; err503=0
for i in $(seq 1 20); do
  code=$(curl -s -o /dev/null -w "%{http_code}" https://glm52-2tp8.jmpti.woa.com/v1/chat/completions \
    -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is 2+2? Just the number."}],"max_tokens":1500,"stream":false}' \
    -m 60)
  case "$code" in
    200) ok=$((ok + 1)) ;;
    503) err503=$((err503 + 1)); echo "  req$i: 503" ;;
    *) fail=$((fail + 1)); echo "  req$i: $code" ;;
  esac
done
echo "Sequential: $ok OK, $err503 503, $fail other"

echo ""
echo "=== Test 2: 3 waves of 20 concurrent requests (60 total) ==="
total_ok=0; total_fail=0; total_503=0
for wave in 1 2 3; do
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
  done
  wait
  wave_ok=0; wave_fail=0; wave_503=0
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
echo "Concurrent: $total_ok OK, $total_503 503, $total_fail other (out of 60)"

echo ""
echo "=== Test 3: 50 sequential requests (stress) ==="
ok=0; fail=0; err503=0
for i in $(seq 1 50); do
  code=$(curl -s -o /dev/null -w "%{http_code}" https://glm52-2tp8.jmpti.woa.com/v1/chat/completions \
    -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}],"max_tokens":50,"stream":false}' \
    -m 60)
  case "$code" in
    200) ok=$((ok + 1)) ;;
    503) err503=$((err503 + 1)); echo "  req$i: 503" ;;
    *) fail=$((fail + 1)); echo "  req$i: $code" ;;
  esac
done
echo "Stress: $ok OK, $err503 503, $fail other (out of 50)"

echo ""
echo "=== Post-test metrics ==="
kubectl exec -n kube-system sglang-glm52-2tp8-router-697df7c955-2zrcc -- curl -s http://127.0.0.1:29000/metrics -m 10 2>&1 | grep -E "^smg_http_responses_total|^smg_router_upstream_responses_total|^smg_worker_cb_outcomes_total|^smg_worker_cb_state" | sort
