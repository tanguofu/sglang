#!/usr/bin/env bash
# Launch PD router (run on bm1 or any CPU node reachable from clients).
# Usage:
#   ./start_router.sh
source "$(dirname "$0")/common.sh"

CONTAINER_NAME="${ROUTER_NAME:-glm52_router}"

echo "[router] prefill=http://$PREFILL_IP:$PREFILL_PORT  decode=http://$DECODE_IP:$DECODE_PORT  port=$ROUTER_PORT"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run -d --network=host \
  --name "$CONTAINER_NAME" \
  "$IMAGE" \
  python3 -m sglang_router.launch_router \
    --pd-disaggregation --mini-lb \
    --prefill "http://$PREFILL_IP:$PREFILL_PORT" \
    --decode  "http://$DECODE_IP:$DECODE_PORT" \
    --host 0.0.0.0 --port "$ROUTER_PORT"

echo "[router] container=$CONTAINER_NAME started. endpoint: http://0.0.0.0:$ROUTER_PORT"
