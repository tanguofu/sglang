#!/bin/bash
# Test /health behavior — idle vs during generation
echo "=== Idle /health ==="
for i in 1 2 3; do
  kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- curl -s http://127.0.0.1:30000/health -m 30 -w " HTTP:%{http_code} t:%{time_total}s\n" -o /dev/null 2>&1
done

echo ""
echo "=== /health during concurrent generation ==="
kubectl exec -n kube-system sglang-glm52-2tp8-sglang-0 -- bash -c '
  # Launch 3 long generation requests in background
  for j in 1 2 3; do
    curl -s http://127.0.0.1:30000/v1/chat/completions \
      -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"glm-5.2\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a long essay about topic $j\"}],\"max_tokens\":1500,\"stream\":false}" \
      -m 90 > /tmp/long_req_$j.out 2>&1 &
  done
  sleep 2
  echo "Now testing /health while 3 generations are running..."
  for i in 1 2 3 4 5 6 7 8; do
    curl -s http://127.0.0.1:30000/health -m 30 -w " HTTP:%{http_code} t:%{time_total}s\n" -o /dev/null 2>&1
  done
  wait
  echo "Generation requests completed"
'
