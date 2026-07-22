#!/bin/bash
# Waits for test32-sglang-0 READY, copies scripts, runs cold 5x hicache test.
set -uo pipefail
NS=kube-system
POD=test32-sglang-0
APIKEY=sk-46faecc9d0bc4dcd9db6a15c73ae91c8

echo "waiting for READY (up to 22 min)..."
ready=0
for i in $(seq 1 264); do
  sleep 5
  st=$(kubectl -n $NS get pod $POD -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)
  rc=$(kubectl -n $NS get pod $POD -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null)
  if [ "$st" = "true" ]; then ready=1; break; fi
  if [ -n "$rc" ] && [ "$rc" != "0" ]; then
    echo "RESTART detected (restartCount=$rc) at iter $i"
    kubectl -n $NS logs $POD --tail=20 2>&1 | tail -20
  fi
  if [ $((i % 12)) -eq 0 ]; then echo "  ...iter $i ready=$st rc=$rc"; fi
done
if [ "$ready" != "1" ]; then
  echo "FAILED: not READY after 22 min"
  kubectl -n $NS logs $POD --tail=40 2>&1 | tail -40
  exit 1
fi
rc=$(kubectl -n $NS get pod $POD -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null)
echo "READY, restartCount=$rc"

# Verify write policy = write_back
args=$(kubectl -n $NS get pod $POD -o jsonpath='{.spec.containers[0].args}')
echo "$args" | grep -q -- "--hicache-write-policy write_back " && echo "policy=write_back OK" || echo "WARN: policy not write_back"
echo "$args" | grep -oE -- "--speculative-num-steps [0-9]+ --speculative-num-draft-tokens [0-9]+ --speculative-eagle-topk [0-9]+"

# Copy scripts into fresh container
kubectl -n $NS cp /tmp/test32_stress_v2.py $POD:/tmp/test32_stress_v2.py >/dev/null 2>&1
kubectl -n $NS cp /tmp/hicache_test.py $POD:/tmp/hicache_test.py >/dev/null 2>&1
kubectl -n $NS exec $POD -- python3 -c "import ast; ast.parse(open('/tmp/hicache_test.py').read())" >/dev/null 2>&1 && echo "scripts copied OK" || echo "WARN script copy"

echo "=== COLD write_back 5x repeated-prefix hicache test ==="
kubectl -n $NS exec $POD -- python3 /tmp/hicache_test.py --ctx-tokens 12000 --max-tokens 256 --repeats 5 2>&1 | tee /tmp/hicache_writeback_cold.log

echo "=== final metrics ==="
kubectl -n $NS exec $POD -- curl -s -H "Authorization: Bearer $APIKEY" localhost:30000/metrics 2>&1 | grep -E "^sglang:(hicache_host_used_tokens|hicache_host_total_tokens|hicache_gpu_used_tokens|cache_hit_rate)\b"
rc2=$(kubectl -n $NS get pod $POD -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null)
echo "final restartCount=$rc2"
echo "===== COLD write_back DONE ====="
