#!/bin/bash
# Patch service selectors to add app.kubernetes.io/instance label
# This prevents test pods from being matched by production service selectors

echo "=== Current selectors (before patch) ==="
for svc in sglang-glm52-2tp8-router sglang-glm52-2tp8-sglang sglang-glm52-2tp8-sglang-headless sglang-glm52-2tp8-w2-sglang sglang-glm52-2tp8-w2-sglang-headless; do
  echo -n "$svc: "
  kubectl get svc "$svc" -n kube-system -o jsonpath='{.spec.selector}' 2>&1
  echo ""
done

echo ""
echo "=== Patching sglang-glm52-2tp8-router (add instance=sglang-glm52-2tp8) ==="
# This service should only match production router (instance=sglang-glm52-2tp8)
kubectl patch svc sglang-glm52-2tp8-router -n kube-system --type=json -p='[{"op":"add","path":"/spec/selector/app.kubernetes.io~1instance","value":"sglang-glm52-2tp8"}]' 2>&1

echo ""
echo "=== Patching sglang-glm52-2tp8-sglang (add instance=sglang-glm52-2tp8) ==="
# This service should only match production worker 1 (instance=sglang-glm52-2tp8)
kubectl patch svc sglang-glm52-2tp8-sglang -n kube-system --type=json -p='[{"op":"add","path":"/spec/selector/app.kubernetes.io~1instance","value":"sglang-glm52-2tp8"}]' 2>&1

echo ""
echo "=== Patching sglang-glm52-2tp8-sglang-headless (add instance=sglang-glm52-2tp8) ==="
# This service should only match production worker 1 (instance=sglang-glm52-2tp8)
kubectl patch svc sglang-glm52-2tp8-sglang-headless -n kube-system --type=json -p='[{"op":"add","path":"/spec/selector/app.kubernetes.io~1instance","value":"sglang-glm52-2tp8"}]' 2>&1

echo ""
echo "=== Patching sglang-glm52-2tp8-w2-sglang (add instance=sglang-glm52-2tp8-w2) ==="
# This service should only match production worker 2 (instance=sglang-glm52-2tp8-w2)
kubectl patch svc sglang-glm52-2tp8-w2-sglang -n kube-system --type=json -p='[{"op":"add","path":"/spec/selector/app.kubernetes.io~1instance","value":"sglang-glm52-2tp8-w2"}]' 2>&1

echo ""
echo "=== Patching sglang-glm52-2tp8-w2-sglang-headless (add instance=sglang-glm52-2tp8-w2) ==="
# This service should only match production worker 2 (instance=sglang-glm52-2tp8-w2)
kubectl patch svc sglang-glm52-2tp8-w2-sglang-headless -n kube-system --type=json -p='[{"op":"add","path":"/spec/selector/app.kubernetes.io~1instance","value":"sglang-glm52-2tp8-w2"}]' 2>&1

echo ""
echo "=== New selectors (after patch) ==="
for svc in sglang-glm52-2tp8-router sglang-glm52-2tp8-sglang sglang-glm52-2tp8-sglang-headless sglang-glm52-2tp8-w2-sglang sglang-glm52-2tp8-w2-sglang-headless; do
  echo -n "$svc: "
  kubectl get svc "$svc" -n kube-system -o jsonpath='{.spec.selector}' 2>&1
  echo ""
done

echo ""
echo "=== New endpoints (after patch) ==="
for svc in sglang-glm52-2tp8-router sglang-glm52-2tp8-sglang sglang-glm52-2tp8-w2-sglang; do
  echo -n "$svc: "
  kubectl get endpoints "$svc" -n kube-system -o jsonpath='{.subsets[*].addresses[*].ip}' 2>&1
  echo ""
done
