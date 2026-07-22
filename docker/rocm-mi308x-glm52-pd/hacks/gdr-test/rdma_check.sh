#!/bin/bash
# Measure RDMA traffic for single request
echo "=== Baseline (before request) ==="
kubectl exec -n kube-system sglang-1p1d-prefill-0 -- bash -c '
for i in 0 1 2 3 4 5 6 7; do
  tx=$(cat /sys/class/net/bond$i/statistics/tx_bytes 2>/dev/null)
  printf "bond%s_tx:%s " "$i" "$tx"
done
echo
'

echo ""
echo "=== Sending 1 request (max_tokens=200, ~200 word essay) ==="
kubectl exec -n kube-system deploy/sglang-1p1d-router -- curl -s -w "\ntime_total: %{time_total}s\n" -o /dev/null --max-time 30 \
  -X POST http://localhost:30011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-46faecc9d0bc4dcd9db6a15c73ae91c8" \
  -d '{"model":"default","max_tokens":200,"messages":[{"role":"user","content":"Write a 200 word essay about artificial intelligence and its impact on society"}]}'

echo ""
echo "=== After request ==="
kubectl exec -n kube-system sglang-1p1d-prefill-0 -- bash -c '
for i in 0 1 2 3 4 5 6 7; do
  tx=$(cat /sys/class/net/bond$i/statistics/tx_bytes 2>/dev/null)
  printf "bond%s_tx:%s " "$i" "$tx"
done
echo
'

echo ""
echo "=== Decode side RX ==="
kubectl exec -n kube-system sglang-1p1d-decode-0 -- bash -c '
for i in 0 1 2 3 4 5 6 7; do
  rx=$(cat /sys/class/net/bond$i/statistics/rx_bytes 2>/dev/null)
  printf "bond%s_rx:%s " "$i" "$rx"
done
echo
'
