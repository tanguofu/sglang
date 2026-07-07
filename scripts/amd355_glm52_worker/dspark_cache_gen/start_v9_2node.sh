#!/bin/bash
# 2-node dspark v9 training (RDMA + Docker + mp.spawn)
# Usage: bash start_v9_2node.sh <node_rank>
#   node-2 = 0 (master)
#   node-3 = 1
set -e

NODE_RANK=${1:?Usage: bash $0 <node_rank 0-1>}
MASTER_ADDR=66.42.112.222
MASTER_PORT=29500
IMAGE=lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260704
NNODES=2
NPROC_PER_NODE=8
WORLD_SIZE=16
CONFIG="config/dspark/dspark_glm5_2_v9_clean.py"
LOG_FILE="/data/v9_2node_train.log"
CONTAINER_NAME="glm52_dspark_v9_2node"
GID_INDEX=1

echo "[$(date)] Launching dspark v9 2-node training (mp.spawn mode)"
echo "  node_rank=$NODE_RANK  gid_index=$GID_INDEX  master=$MASTER_ADDR:$MASTER_PORT"
echo "  nnodes=$NNODES  nproc=$NPROC_PER_NODE  world_size=$WORLD_SIZE"

docker rm -f ${CONTAINER_NAME} 2>/dev/null || true
sleep 1

docker run -d --name ${CONTAINER_NAME} \
  --device /dev/kfd --device /dev/dri --network host --shm-size 64G --ipc host \
  --privileged -v /data:/data \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e SGLANG_USE_AITER=1 \
  -e USE_TORCH=true \
  -e WANDB_DISABLED=true \
  -e TOKENIZERS_PARALLELISM=false \
  -e PYTORCH_ROCM_ARCH="gfx942;gfx950" \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT8 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HSA_ENABLE_SDMA=0 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=ionic \
  -e NCCL_IB_GID_INDEX=${GID_INDEX} \
  -e NCCL_NET_GDR_LEVEL=0 \
  -e NCCL_MIN_NCHANNELS=16 \
  -e NCCL_SOCKET_IFNAME=enp193s0f0np0 \
  -e NCCL_P2P_DISABLE=0 \
  -e NCCL_DEBUG=WARN \
  -e NCCL_TIMEOUT=1200 \
  -e GLOO_SOCKET_IFNAME=enp193s0f0np0 \
  -e MASTER_ADDR=${MASTER_ADDR} \
  -e MASTER_PORT=${MASTER_PORT} \
  -e NODE_RANK=${NODE_RANK} \
  -e WORLD_SIZE=${WORLD_SIZE} \
  -e NNODES=${NNODES} \
  -e NPROC_PER_NODE=${NPROC_PER_NODE} \
  "$IMAGE" \
  bash -c 'set -e; \
    pip install -q tensorboard 2>/dev/null; \
    cd /usr/lib/x86_64-linux-gnu/libibverbs && ln -sf libionic-rdmav34.so libbnxt_re-rdmav34.so 2>/dev/null; ldconfig 2>/dev/null; \
    cd /data/DeepSpec && \
    python3 run_v9_manual.py \
      --config '"${CONFIG}"' \
      2>&1 | tee '"${LOG_FILE}"''

echo "[$(date)] Container ${CONTAINER_NAME} launched (rank=$NODE_RANK)"
echo "  Log: ${LOG_FILE}"
