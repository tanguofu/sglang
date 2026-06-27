#!/bin/bash
# Sequential PD startup inside one container (avoid mori RegEndpoint race).
# Single-node KV transfer: mori over XGMI (not ionic RoCE RDMA).
set -euo pipefail
SCHEME=${SCHEME:-PD-1a}
BACKEND=${BACKEND:-mori}
source /data/pd_single_node/common.sh

P_PP=${P_PP:-1}; P_TP=${P_TP:-4}; D_PP=${D_PP:-1}; D_TP=${D_TP:-4}; D_BASE=${D_BASE:-4}
P_PATCH=${P_PATCH:-"python3 /data/patch_glm_config.py 2>/dev/null || true"}
D_PATCH=${D_PATCH:-"python3 /data/patch_glm_config.py 2>/dev/null || true"}
D_SPEC=${D_SPEC:-""}

export MORI_DISABLE_AUTO_XGMI=0
export MORI_IO_NODE_ID=${MORI_IO_NODE_ID:-mi355x-single-node}
export MORI_RDMA_DEVICES="^ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
unset SGLANG_HOST_IP HOST_IP

eval "${PP_ENV:-true}"
eval "$P_PATCH"
eval "$D_PATCH"

COMMON="--model-path $MODEL_PATH --trust-remote-code --host 0.0.0.0 --context-length $CONTEXT_LENGTH --tool-call-parser glm47 --reasoning-parser glm45 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 --chunked-prefill-size 32768 --enable-fused-qk-norm-rope --watchdog-timeout 3600 --log-level info --disaggregation-transfer-backend $BACKEND --disaggregation-bootstrap-port $BOOTSTRAP_PORT"

echo "[PD] Starting prefill (XGMI mori, no RDMA ib-device)..."
python3 -m sglang.launch_server $COMMON --disaggregation-mode prefill --tp-size $P_TP --pp-size $P_PP --port $PREFILL_PORT --dist-init-addr 127.0.0.1:29600 --max-running-requests 32 --enable-aiter-allreduce-fusion \
  > /data/pd_single_node/logs/${SCHEME}_${BACKEND}_prefill.log 2>&1 &
P_PID=$!

for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:${PREFILL_PORT}/health" >/dev/null 2>&1 && break
  sleep 10
done
curl -sf "http://127.0.0.1:${PREFILL_PORT}/health" || { echo prefill failed; tail -30 "/data/pd_single_node/logs/${SCHEME}_${BACKEND}_prefill.log"; exit 1; }
echo "[PD] Prefill ready, starting decode (XGMI mori)..."

# NCCL_P2P_DISABLE avoids hipIpcGetMemHandle failures when P/D share /dev/dri.
export NCCL_P2P_DISABLE=1
python3 -m sglang.launch_server $COMMON --disaggregation-mode decode --tp-size $D_TP --pp-size $D_PP --base-gpu-id $D_BASE --port $DECODE_PORT --dist-init-addr 127.0.0.1:29700 --disable-radix-cache --max-running-requests 64 $D_SPEC \
  > /data/pd_single_node/logs/${SCHEME}_${BACKEND}_decode.log 2>&1 &
D_PID=$!

for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:${DECODE_PORT}/health" >/dev/null 2>&1 && break
  sleep 10
done
curl -sf "http://127.0.0.1:${DECODE_PORT}/health" || { echo decode failed; tail -30 "/data/pd_single_node/logs/${SCHEME}_${BACKEND}_decode.log"; exit 1; }
echo "[PD] Decode ready, starting router..."

python3 -m sglang_router.launch_router --pd-disaggregation --mini-lb \
  --prefill "http://127.0.0.1:$PREFILL_PORT" --decode "http://127.0.0.1:$DECODE_PORT" \
  --host 0.0.0.0 --port $ROUTER_PORT \
  > /data/pd_single_node/logs/${SCHEME}_${BACKEND}_router.log 2>&1 &
R_PID=$!

for i in $(seq 1 30); do
  curl -sf "http://127.0.0.1:${ROUTER_PORT}/health" >/dev/null 2>&1 && break
  sleep 2
done
echo "[PD] Stack ready: P=$P_PID D=$D_PID R=$R_PID"
echo "P=$P_PID D=$D_PID R=$R_PID" > "/data/pd_single_node/logs/${SCHEME}_${BACKEND}_pids.txt"

while kill -0 $P_PID 2>/dev/null && kill -0 $D_PID 2>/dev/null; do sleep 30; done
echo "[PD] Child died"
exit 1
