#!/bin/bash
# Test RDMA connectivity for all 8 IB devices
DECODE_IPS=(
  "26.24.42.106"   # bond0
  "26.24.44.166"   # bond1
  "26.24.43.82"    # bond2
  "26.24.43.254"   # bond3
  "26.24.46.190"   # bond4
  "26.24.45.214"   # bond5
  "26.24.46.90"    # bond6
  "26.24.45.106"   # bond7
)

for i in 0 1 2 3 4 5 6 7; do
  PORT=$((18515 + i))
  DEV="bnxt_re_bond${i}"
  DST="${DECODE_IPS[$i]}"
  echo "=== bond${i}: ${DEV} -> ${DST}:${PORT} ==="
  kubectl exec -n kube-system sglang-1p1d-decode-0 -- bash -c "nohup ib_write_bw -d ${DEV} -p ${PORT} --report_gbits > /tmp/ib_test_bond${i}.log 2>&1 &" 2>&1
  sleep 1
  kubectl exec -n kube-system sglang-1p1d-prefill-0 -- bash -c "timeout 30 ib_write_bw -d ${DEV} -p ${PORT} ${DST} --report_gbits 2>&1" 2>&1 | grep -E "BW average|error|fail|Connection" | head -3
  echo ""
done
