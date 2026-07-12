#!/usr/bin/env bash
# Shared config for GLM-5.2-FP8 1P1D disaggregated serving on AMD MI355X (2-node).
# Backends: mooncake (default) or mori. Both use RDMA over ionic (Pensando RoCE v2).
#
# Override any variable via environment before sourcing/running.

set -euo pipefail

# --- Image & model ---
IMAGE="${IMAGE:-sglang-glm52-mi355x-pd:latest}"
MODEL_PATH="${MODEL_PATH:-/data/models/GLM-5.2-FP8}"

# --- Topology: bm1=prefill, bm2=decode (each TP8, single-node-per-role) ---
PREFILL_IP="${PREFILL_IP:-216.128.154.57}"     # amd-bare-metal-1
DECODE_IP="${DECODE_IP:-149.28.114.238}"       # amd-bare-metal-2
PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT="${DECODE_PORT:-30001}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
ROUTER_PORT="${ROUTER_PORT:-8000}"

# --- RDMA devices (8x AMD Pensando RoCE v2: ionic_0..ionic_7, 400Gbps) ---
# NOTE: ionic_2 is DOWN on bare-metal-1 → prefill excludes it by default.
PREFILL_IB="${PREFILL_IB:-ionic_0,ionic_1,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}"
DECODE_IB="${DECODE_IB:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}"

# --- Transfer backend: mooncake | mori ---
BACKEND="${BACKEND:-mooncake}"

# Per-backend GID / env. ionic interfaces are IPv6-only; the two nodes share
# the fd93:16d3:59b6::/64 prefix so RoCE GIDs are mutually reachable.
#   mooncake: MC_GID_INDEX=1  (global IPv6 GID, from team DSv4 PD scripts)
#   mori:     NCCL_IB_GID_INDEX=3 + MORI_DISABLE_AUTO_XGMI=1 (from team GLM PD scripts)
case "$BACKEND" in
  mooncake)
    GID_INDEX_ENV=(-e MC_GID_INDEX="${MC_GID_INDEX:-1}")
    ;;
  mori)
    # MORI_DISABLE_AUTO_XGMI=1 disables intra-node XGMI (no XGMI link across machines)
    GID_INDEX_ENV=(-e NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
                   -e MORI_DISABLE_AUTO_XGMI=1)
    ;;
  *)
    echo "ERROR: unknown BACKEND=$BACKEND (use 'mooncake' or 'mori')" >&2
    exit 1
    ;;
esac

# --- Shared sglang server args (prefill & decode) ---
COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --trust-remote-code
  --host 0.0.0.0
  --context-length 1048576
  --tool-call-parser glm47
  --reasoning-parser glm45
  --kv-cache-dtype fp8_e4m3
  --mem-fraction-static 0.85
  --chunked-prefill-size 32768
  --enable-fused-qk-norm-rope
  --watchdog-timeout 3600
  --log-level info
  --tp 8
  --disaggregation-transfer-backend "$BACKEND"
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
  --enable-cache-report
)

# --- Shared docker run flags ---
DOCKER_FLAGS=(
  -d --privileged --network=host --ipc=host
  --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --security-opt seccomp=unconfined
  --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband
  --ulimit memlock=-1 --shm-size 64g --group-add video
  -v /data/:/data/
  -v /sys/class/infiniband:/sys/class/infiniband:ro
  -v /sys/class/net:/sys/class/net:ro
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  # ROCm cross-node: ionic RoCE doesn't support GPUDirect RDMA to AMD GPU memory.
  # Stage KV GPU->host (hipMemcpy) -> RDMA transfer on host memory -> host->GPU.
  # Keeps the RDMA data path (over ionic) while avoiding GPU-direct RDMA.
  -e SGLANG_PD_HOST_STAGING=1
)
