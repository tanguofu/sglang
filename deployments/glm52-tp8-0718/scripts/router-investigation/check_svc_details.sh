#!/bin/bash
echo "=== sglang-glm52-2tp8-sglang-headless selector ==="
kubectl get svc sglang-glm52-2tp8-sglang-headless -n kube-system -o jsonpath='{.spec.selector}' 2>&1 | python3 -m json.tool
echo ""
echo "=== sglang-glm52-2tp8-w2-sglang-headless selector ==="
kubectl get svc sglang-glm52-2tp8-w2-sglang-headless -n kube-system -o jsonpath='{.spec.selector}' 2>&1 | python3 -m json.tool
echo ""
echo "=== sglang-glm52-2tp8-sglang service full yaml ==="
kubectl get svc sglang-glm52-2tp8-sglang -n kube-system -o yaml 2>&1
echo ""
echo "=== sglang-glm52-2tp8-w2-sglang service full yaml ==="
kubectl get svc sglang-glm52-2tp8-w2-sglang -n kube-system -o yaml 2>&1
