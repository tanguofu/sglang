#!/bin/bash
# =============================================================================
# DSpark GLM-5.2 v9 CLEAN — 4-node (32 GPU) RDMA training launcher
# =============================================================================
# Usage: bash start_v9_4node_clean.sh <node_rank>
#   0 = node-2  (66.42.112.222, master)
#   1 = node-4  (149.28.124.220)
#   2 = node-9  (104.207.141.239, GID_INDEX=3!)
#   3 = worker  (144.202.61.0, amd-355-worker)
#
# RDMA config reference: iWiki 4024263928
#   NCCL_P2P_DISABLE=0  (XGMI P2P intra-node, 3.5x speedup)
#   NCCL_NET_GDR_LEVEL=0  (avoid bootstrap conflict)
#   NCCL_IB_GID_INDEX=1  (node-9 uses 3, ULA IPv6 GID required)
#   NCCL_IB_HCA=ionic
#   NCCL_MIN_NCHANNELS=16
#
# Prerequisites (run sync_4node_code.sh + sync_4node_cache.sh first):
#   - /data/dspark_target_cache_v9_coding_clean_merged (114G)
#   - /data/DeepSpec with all fixes (parser, modeling, nan_to_num, ckpt, base_trainer)
#   - /data/DeepSpec/config/dspark/dspark_glm5_2_v9_clean_4node.py
# =============================================================================

set -e

NODE_RANK=${1:?Usage: bash $0 <node_rank 0-3>}

MASTER_ADDR=66.42.112.222
MASTER_PORT=29500
IMAGE=lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260704
NNODES=4
NPROC_PER_NODE=8
WORLD_SIZE=32
CONFIG="config/dspark/dspark_glm5_2_v9_clean_4node.py"
LOG_FILE="/data/v9_4node_clean_train.log"
CONTAINER_NAME="glm52_dspark_v9_4node_clean"

# Per-node GID_INDEX (node-9 needs 3, others use 1)
GID_INDEX=1
if [ "$NODE_RANK" = "2" ]; then
    GID_INDEX=3   # node-9: gid[1] is IPv4-mapped, gid[3] is ULA IPv6
fi

# Peer IPs for firewall rules
ALL_PEER_IPS="66.42.112.222 149.28.124.220 104.207.141.239 144.202.61.0"
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "[$(date)] Launching DSpark v9 CLEAN 4-node training (mp.spawn mode)"
echo "  node_rank=$NODE_RANK  master=$MASTER_ADDR:$MASTER_PORT"
echo "  nnodes=$NNODES  nproc=$NPROC_PER_NODE  world_size=$WORLD_SIZE"
echo "  gid_index=$GID_INDEX  config=$CONFIG"
echo "  cpu_offload=OFF (DS_CPU_OFFLOAD=0)"

# Open firewall for all peer nodes
for peer_ip in $ALL_PEER_IPS; do
    if [ "$peer_ip" != "$LOCAL_IP" ]; then
        iptables -C INPUT -s "$peer_ip" -j ACCEPT 2>/dev/null || \
            iptables -I INPUT -s "$peer_ip" -j ACCEPT 2>/dev/null || true
    fi
done

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
  \
  -e DS_CPU_OFFLOAD=0 \
  \
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
  \
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
echo "  Monitor: docker logs -f ${CONTAINER_NAME}"
