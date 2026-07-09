#!/usr/bin/env bash
# docker-run.sh — convenience script to launch the master-snapshot image.
# Replicates the exact docker run flags from start_worker_0706_no_fused.sh.
set -euo pipefail

IMAGE=${IMAGE:-sglang-glm52-master-snapshot:latest}
CONTAINER_NAME=${CONTAINER_NAME:-sglang_master_snapshot}
PORT=${PORT:-30000}

# Remove existing container if REPLACE=1
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  if [[ "${REPLACE:-0}" == "1" ]]; then
    docker rm -f "$CONTAINER_NAME"
  else
    echo "ERROR: container $CONTAINER_NAME already exists. Set REPLACE=1 to remove." >&2
    exit 1
  fi
fi

echo "Starting $CONTAINER_NAME from $IMAGE on port $PORT"

docker run -d \
  --name "$CONTAINER_NAME" \
  --privileged \
  --network host \
  --shm-size 32g \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /data:/data \
  "$IMAGE"

echo "Started. Check logs with: docker logs -f $CONTAINER_NAME"
