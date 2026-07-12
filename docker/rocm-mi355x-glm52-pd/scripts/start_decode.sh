#!/usr/bin/env bash
# Launch GLM-5.2-FP8 decode worker (run on bm2 = 149.28.114.238).
# Usage:
#   ./start_decode.sh                 # default backend=mooncake
#   BACKEND=mori ./start_decode.sh    # use mori backend
source "$(dirname "$0")/common.sh"

CONTAINER_NAME="${DECODE_NAME:-glm52_decode}"

echo "[decode] backend=$BACKEND  ib=$DECODE_IB  port=$DECODE_PORT  bootstrap→$PREFILL_IP:$BOOTSTRAP_PORT"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run "${DOCKER_FLAGS[@]}" \
  "${GID_INDEX_ENV[@]}" \
  --name "$CONTAINER_NAME" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    "${COMMON_ARGS[@]}" \
    --port "$DECODE_PORT" \
    --dist-init-addr 127.0.0.1:29700 \
    --disaggregation-mode decode \
    --disaggregation-ib-device "$DECODE_IB" \
    --disable-radix-cache \
    --max-running-requests 64

echo "[decode] container=$CONTAINER_NAME started. logs: docker logs -f $CONTAINER_NAME"
