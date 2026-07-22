#!/bin/bash
# Compare PD vs TP8 configurations in detail
echo "============================================================"
echo "  PD vs TP8 Configuration Comparison"
echo "============================================================"

echo ""
echo "=== PD Prefill 启动参数 ==="
kubectl exec -n kube-system sglang-1p1d-prefill-0 -- bash -c "ps aux | grep 'python3 -m sglang' | grep -v grep" | tr ' ' '\n' | grep -E "^--" > /tmp/pd_prefill_args.txt
cat /tmp/pd_prefill_args.txt

echo ""
echo "=== PD Decode 启动参数 ==="
kubectl exec -n kube-system sglang-1p1d-decode-0 -- bash -c "ps aux | grep 'python3 -m sglang' | grep -v grep" | tr ' ' '\n' | grep -E "^--" > /tmp/pd_decode_args.txt
cat /tmp/pd_decode_args.txt

echo ""
echo "=== TP8 Worker 启动参数 ==="
kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- bash -c "ps aux | grep 'python3 -m sglang' | grep -v grep" | tr ' ' '\n' | grep -E "^--" > /tmp/tp8_args.txt
cat /tmp/tp8_args.txt

echo ""
echo "=== 关键差异对比 ==="
echo ""
printf "%-35s %-20s %-20s %-20s\n" "Parameter" "PD Prefill" "PD Decode" "TP8 Worker"
echo "---------------------------------------------------------------------------------------------------"

params=(
    "mem-fraction-static"
    "chunked-prefill-size"
    "schedule-conservativeness"
    "prefill-max-requests"
    "max-prefill-tokens"
    "max-running-requests"
    "speculative-algorithm"
    "speculative-num-steps"
    "speculative-num-draft-tokens"
    "speculative-eagle-topk"
    "cuda-graph-bs-prefill"
    "cuda-graph-bs-decode"
    "cuda-graph-max-bs-decode"
    "cuda-graph-backend-prefill"
    "enable-mixed-chunk"
    "enable-hierarchical-cache"
    "hicache-ratio"
    "disable-overlap-schedule"
    "enable-aiter-allreduce-fusion"
    "enable-fused-qk-norm-rope"
    "num-reserved-decode-tokens"
)

for p in "${params[@]}"; do
    pd_pre=$(grep -A1 "^--${p}$" /tmp/pd_prefill_args.txt 2>/dev/null | tail -1 | grep -v "^--" || echo "N/A")
    pd_dec=$(grep -A1 "^--${p}$" /tmp/pd_decode_args.txt 2>/dev/null | tail -1 | grep -v "^--" || echo "N/A")
    tp8=$(grep -A1 "^--${p}$" /tmp/tp8_args.txt 2>/dev/null | tail -1 | grep -v "^--" || echo "N/A")
    printf "%-35s %-20s %-20s %-20s\n" "$p" "$pd_pre" "$pd_dec" "$tp8"
done

echo ""
echo "=== Mooncake 环境变量 ==="
echo "PD Prefill:"
kubectl exec -n kube-system sglang-1p1d-prefill-0 -- bash -c "env | grep -iE 'MC_|MOONCAKE' | sort"
echo ""
echo "PD Decode:"
kubectl exec -n kube-system sglang-1p1d-decode-0 -- bash -c "env | grep -iE 'MC_|MOONCAKE' | sort"
