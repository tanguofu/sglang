#!/bin/bash
# Check all 2tp8 service endpoints
echo "=== sglang-glm52-2tp8-router endpoints ==="
kubectl get endpoints sglang-glm52-2tp8-router -n kube-system -o jsonpath='{.subsets}' 2>&1 | python3 -c "
import json, sys
data = json.load(sys.stdin)
for s in data:
    for addr in s.get('addresses', []):
        ref = addr.get('targetRef', {})
        print(f\"  {addr.get('ip')} -> {ref.get('name')} (instance: see labels)\")
"

echo ""
echo "=== sglang-glm52-2tp8-sglang endpoints ==="
kubectl get endpoints sglang-glm52-2tp8-sglang -n kube-system -o jsonpath='{.subsets}' 2>&1 | python3 -c "
import json, sys
data = json.load(sys.stdin)
for s in data:
    for addr in s.get('addresses', []):
        ref = addr.get('targetRef', {})
        print(f\"  {addr.get('ip')} -> {ref.get('name')}\")
"

echo ""
echo "=== sglang-glm52-2tp8-w2-sglang endpoints ==="
kubectl get endpoints sglang-glm52-2tp8-w2-sglang -n kube-system -o jsonpath='{.subsets}' 2>&1 | python3 -c "
import json, sys
data = json.load(sys.stdin)
for s in data:
    for addr in s.get('addresses', []):
        ref = addr.get('targetRef', {})
        print(f\"  {addr.get('ip')} -> {ref.get('name')}\")
"

echo ""
echo "=== sglang-glm52-2tp8-sglang-headless endpoints ==="
kubectl get endpoints sglang-glm52-2tp8-sglang-headless -n kube-system -o jsonpath='{.subsets}' 2>&1 | python3 -c "
import json, sys
data = json.load(sys.stdin)
for s in data:
    for addr in s.get('addresses', []):
        ref = addr.get('targetRef', {})
        print(f\"  {addr.get('ip')} -> {ref.get('name')}\")
"

echo ""
echo "=== sglang-glm52-2tp8-w2-sglang-headless endpoints ==="
kubectl get endpoints sglang-glm52-2tp8-w2-sglang-headless -n kube-system -o jsonpath='{.subsets}' 2>&1 | python3 -c "
import json, sys
data = json.load(sys.stdin)
for s in data:
    for addr in s.get('addresses', []):
        ref = addr.get('targetRef', {})
        print(f\"  {addr.get('ip')} -> {ref.get('name')}\")
"
