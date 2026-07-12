#!/usr/bin/env bash
# Clean up all PD RDMA pods and kill GPU processes on both ZW nodes.
# Usage: bash clean-gpu.sh

set -euo pipefail

echo "=== Deleting old PD pods ==="
kubectl delete pod -n kube-system pd-prefill-rdma pd-decode-rdma pd-router-rdma --force --grace-period=0 2>/dev/null || true
kubectl delete pod -n kube-system pd-prefill-rdma6 pd-decode-rdma6 pd-router-rdma6 --force --grace-period=0 2>/dev/null || true

echo ""
echo "=== Killing GPU processes on ZW nodes ==="
for NODE in node-21.234.170.19 node-21.234.170.32; do
    echo "--- $NODE ---"
    kubectl debug node/$NODE -it --image=ubuntu -- chroot /host bash -c '
        pkill -9 -f sglang 2>/dev/null || true
        pkill -9 -f python 2>/dev/null || true
        echo "killed"
    ' 2>&1 | grep -v "^--profile\|^Creating\|^Warning\|^Unable\|^All commands\|^If you\|^$"
done

echo ""
echo "=== Cleanup complete ==="
