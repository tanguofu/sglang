#!/usr/bin/env bash
# Launch GLM-5.2-FP8 prefill worker (run on bm1 = 216.128.154.57).
# Usage:
#   ./start_prefill.sh                 # default backend=mori (cross-node RDMA)
#   BACKEND=mooncake ./start_prefill.sh  # only on mlx5 HCAs
source "$(dirname "$0")/common.sh"

CONTAINER_NAME="${PREFILL_NAME:-glm52_prefill}"

echo "[prefill] backend=$BACKEND  ib=$PREFILL_IB  port=$PREFILL_PORT  bootstrap=$BOOTSTRAP_PORT"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker run "${DOCKER_FLAGS[@]}" \
  "${GID_INDEX_ENV[@]}" \
  --name "$CONTAINER_NAME" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    "${COMMON_ARGS[@]}" \
    --port "$PREFILL_PORT" \
    --dist-init-addr 127.0.0.1:29600 \
    --disaggregation-mode prefill \
    --disaggregation-ib-device "$PREFILL_IB" \
    --max-running-requests 32

echo "[prefill] container=$CONTAINER_NAME started. logs: docker logs -f $CONTAINER_NAME"
