#!/bin/bash
# Verify pod labels for instance distinction
echo "=== Production router pod labels ==="
kubectl get pod -n kube-system sglang-glm52-2tp8-router-697df7c955-2zrcc -o jsonpath='{.metadata.labels}' 2>&1 | python3 -m json.tool
echo ""
echo "=== Test router pod labels ==="
kubectl get pod -n kube-system sglang-glm52-test-router-779db48587-hgd26 -o jsonpath='{.metadata.labels}' 2>&1 | python3 -m json.tool
echo ""
echo "=== Production worker 1 (sglang-0) labels ==="
kubectl get pod -n kube-system sglang-glm52-2tp8-sglang-0 -o jsonpath='{.metadata.labels}' 2>&1 | python3 -m json.tool
echo ""
echo "=== Production worker 2 (w2-sglang-0) labels ==="
kubectl get pod -n kube-system sglang-glm52-2tp8-w2-sglang-0 -o jsonpath='{.metadata.labels}' 2>&1 | python3 -m json.tool
echo ""
echo "=== Test worker 1 (test-sglang-0) labels ==="
kubectl get pod -n kube-system sglang-glm52-test-sglang-0 -o jsonpath='{.metadata.labels}' 2>&1 | python3 -m json.tool
echo ""
echo "=== Test worker 2 (test-w2-sglang-0) labels ==="
kubectl get pod -n kube-system sglang-glm52-test-w2-sglang-0 -o jsonpath='{.metadata.labels}' 2>&1 | python3 -m json.tool
