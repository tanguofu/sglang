#!/bin/bash
# Shared config for single-node GLM-5.2 PD disaggregation on MI355X
export IMAGE="lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260620"
export MODEL_PATH="/data/models/GLM-5.2-FP8"
export BOOTSTRAP_PORT=9000
export PREFILL_PORT=30010
export DECODE_PORT=30020
export ROUTER_PORT=8000
export CONTEXT_LENGTH=1048576
export LOG_DIR="/data/pd_single_node/logs"
mkdir -p "$LOG_DIR"

wait_health() {
  local url=$1 name=$2 timeout=${3:-600}
  for i in $(seq 1 $((timeout/10))); do
    if curl -sf "${url}/health" >/dev/null 2>&1; then
      echo "[OK] ${name} ready at ${url}"
      return 0
    fi
    sleep 10
    echo "  waiting ${name}... ($((i*10))s)"
  done
  echo "[FAIL] ${name} not ready"
  return 1
}
