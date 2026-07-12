#!/usr/bin/env bash
# Deploy PD RDMA disaggregation: prefill + decode + router
# Usage: bash deploy.sh
#
# Prerequisites:
#   - Model files at /data/model/glm52-fp8 on both nodes
#   - Security group allows tcp:10000-13000 between nodes
#   - /dev/infiniband mounted (bnxt_re_bond0-7)
#   - Images pushed to mirrors.tencent.com/ti-platform/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Step 1: Clean up old pods and GPU processes ==="
bash "${SCRIPT_DIR}/clean-gpu.sh"

echo ""
echo "=== Step 2: Deploy prefill (21.234.170.19) ==="
kubectl apply -f "${SCRIPT_DIR}/prefill.yaml"

echo ""
echo "=== Step 3: Deploy decode (21.234.170.32) ==="
kubectl apply -f "${SCRIPT_DIR}/decode.yaml"

echo ""
echo "=== Step 4: Wait for both servers to be ready ==="
echo "Waiting for prefill..."
until kubectl exec pd-prefill-rdma -n kube-system -- python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:13000/health', timeout=5)" 2>/dev/null; do
    echo "  prefill not ready..."
    sleep 10
done
echo "PREFILL READY"

echo "Waiting for decode..."
until kubectl exec pd-decode-rdma -n kube-system -- python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:13000/health', timeout=5)" 2>/dev/null; do
    echo "  decode not ready..."
    sleep 10
done
echo "DECODE READY"

echo ""
echo "=== Step 5: Deploy router ==="
kubectl apply -f "${SCRIPT_DIR}/router.yaml"

echo "Waiting for router..."
until kubectl exec pd-router-rdma -n kube-system -- python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:13002/health', timeout=5)" 2>/dev/null; do
    echo "  router not ready..."
    sleep 5
done
echo "ROUTER READY"

echo ""
echo "=== Step 6: Test inference ==="
bash "${SCRIPT_DIR}/test-inference.sh"

echo ""
echo "=== Deployment complete! ==="
echo "Router endpoint: http://21.234.170.19:13002/v1/chat/completions"
echo "API key: sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
