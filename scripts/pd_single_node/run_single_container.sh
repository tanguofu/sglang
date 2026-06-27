#!/bin/bash
set -euo pipefail
SCHEME=${1:?Usage: run_single_container.sh PD-1a|PD-1b|PD-1d}
BACKEND=${2:-mori}
source /data/pd_single_node/common.sh
bash /data/pd_single_node/stop_all.sh

P_PP=1; P_TP=4; D_PP=1; D_TP=4; D_BASE=4
P_PATCH="python3 /data/patch_glm_config.py 2>/dev/null || true"
D_PATCH="python3 /data/patch_glm_config.py 2>/dev/null || true"
PP_ENV=""; D_SPEC=""

case "$SCHEME" in
  PD-1a) ;;
  PD-1b)
    P_PP=2; P_TP=2; D_PP=2; D_TP=2; D_BASE=4
    P_PATCH="$P_PATCH; python3 /data/patch_pp_missing_layer.py 2>/dev/null || true; python3 /data/patch_mori_pp_kv_slices.py 2>/dev/null || true"
    D_PATCH="$D_PATCH; python3 /data/patch_pp_missing_layer.py 2>/dev/null || true; python3 /data/patch_mori_pp_kv_slices.py 2>/dev/null || true"
    PP_ENV="export SGLANG_PP_LAYER_PARTITION=39,39"
    ;;
  PD-1d)
    D_SPEC="--speculative-algorithm NEXTN --speculative-num-steps 2 --speculative-num-draft-tokens 3 --speculative-eagle-topk 1"
    ;;
  *) echo "Unknown scheme: $SCHEME"; exit 1;;
esac

echo "=== Starting $SCHEME ($BACKEND) single-container XGMI stack ==="
docker run -d --name sglang_pd_stack \
  --privileged --network host --ipc=host --pid=host --shm-size 64g \
  --ulimit memlock=-1 --cap-add=IPC_LOCK --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /data:/data --device /dev/kfd --device /dev/dri --group-add video \
  -e NCCL_DEBUG=WARN -e HSA_ENABLE_SDMA=0 -e SGLANG_USE_AITER=1 \
  -e MORI_DISABLE_AUTO_XGMI=0 \
  -e MORI_IO_NODE_ID=mi355x-single-node \
  -e MORI_RDMA_DEVICES="^ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7" \
  -e SCHEME="$SCHEME" -e BACKEND="$BACKEND" \
  -e P_PP="$P_PP" -e P_TP="$P_TP" -e D_PP="$D_PP" -e D_TP="$D_TP" -e D_BASE="$D_BASE" \
  -e P_PATCH="$P_PATCH" -e D_PATCH="$D_PATCH" -e PP_ENV="$PP_ENV" -e D_SPEC="$D_SPEC" \
  "$IMAGE" bash /data/pd_single_node/start_pd_stack.sh

for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${ROUTER_PORT}/health" >/dev/null 2>&1; then
    echo "[OK] router ready at ${ROUTER_PORT} after $((i*10))s"
    exit 0
  fi
  sleep 10
  echo "  waiting stack... ($((i*10))s)"
done
echo "[FAIL] stack not ready"; docker logs sglang_pd_stack 2>&1 | tail -40; exit 1
