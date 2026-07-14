#!/usr/bin/env bash
# Entrypoint for vLLM PD worker (prefill or decode) with LMCache RDMA
# Branch: 308x-vllm-llcache-1pd
#
# Environment variables:
#   PD_ROLE=prefill|decode
#   MODEL_PATH=/data/model/glm52-fp8
#   PORT=13000
#   HOST_IP=<this node's management IP>
#   PEER_IP=<peer node's management IP>
#   LMCACHE_RDMA_DEVICE=bnxt_re_bond0
#   LMCACHE_RDMA_PORT=52000
#   TENSOR_PARALLEL_SIZE=8
#   GPU_MEMORY_UTILIZATION=0.90
#   MAX_MODEL_LEN=1048576

set -euo pipefail

PD_ROLE="${PD_ROLE:-prefill}"
MODEL_PATH="${MODEL_PATH:-/data/model/glm52-fp8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.2}"
API_KEY="${API_KEY:-sk-46faecc9d0bc4dcd9db6a15c73ae91c8}"
PORT="${PORT:-13000}"
HOST_IP="${HOST_IP:-0.0.0.0}"
PEER_IP="${PEER_IP:-127.0.0.1}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
RDMA_DEVICE="${LMCACHE_RDMA_DEVICE:-bnxt_re_bond0}"
RDMA_PORT="${LMCACHE_RDMA_PORT:-52000}"

echo "============================================"
echo " vLLM PD Worker (${PD_ROLE})"
echo "============================================"
echo " Model: $MODEL_PATH"
echo " Port: $PORT  TP: $TP_SIZE"
echo " Host: $HOST_IP  Peer: $PEER_IP"
echo " RDMA: $RDMA_DEVICE:$RDMA_PORT"

# Determine KV transfer role
if [ "$PD_ROLE" = "prefill" ]; then
    KV_ROLE="kv_producer"
    NIXL_ROLE="sender"
elif [ "$PD_ROLE" = "decode" ]; then
    KV_ROLE="kv_consumer"
    NIXL_ROLE="receiver"
else
    echo "ERROR: Unknown PD_ROLE=$PD_ROLE (expected prefill|decode)"
    exit 1
fi

# KV transfer config — LMCache with NIXL/UCX RDMA
# Correct format from vLLM source: kv_connector/kv_role/kv_connector_extra_config
# NIXL registers CPU pinned memory for RDMA (not GPU Direct, no peer_mem needed)
# UCX_TLS=ib,rdmacm enables RDMA transport (not TCP)
# kv_buffer_device=cpu → host memory bounce buffer (GPU→host→RDMA→host→GPU)
KV_TRANSFER_CONFIG=$(cat <<EOF
{
    "kv_connector": "LMCacheConnectorV1",
    "kv_role": "${KV_ROLE}",
    "kv_connector_extra_config": {
        "use_native": true
    }
}
EOF
)

# LMCache YAML config (NIXL RDMA with host memory)
export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-/etc/lmcache/lmcache-config.yaml}"
mkdir -p /etc/lmcache
cat > "$LMCACHE_CONFIG_FILE" <<LMCACHEYAML
chunk_size: 256
local_cpu: False
max_local_cpu_size: 0
max_local_disk_size: 0
remote_serde: NULL

enable_nixl: True
nixl_role: "${NIXL_ROLE}"
nixl_peer_host: "${PEER_IP}"
nixl_peer_init_port: ${NIXL_PEER_PORT:-55555}
nixl_peer_alloc_port: ${NIXL_PEER_PORT:-55556}
nixl_buffer_size: 1073741824
nixl_buffer_device: "cpu"
nixl_enable_gc: True
LMCACHEYAML

# UCX RDMA transport — use rc_verbs (generic IBverbs) NOT rc_mlx5 (Mellanox-only)
# bnxt_re is a standard IBverbs provider, rc_verbs works with it
# cuda_copy=GPU↔CPU memcpy for host staging, cuda_ipc=GPU IPC for same-node
export UCX_TLS="${UCX_TLS:-rc_verbs,cuda_copy,cuda_ipc,tcp,self}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-bnxt_re_bond0:1}"
# RCCL env vars (for tensor parallel all-reduce, NOT for KV transfer)
export NCCL_IB_HCA="${NCCL_IB_HCA:-bnxt_re}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
# Force rc_verbs over rc_mlx5 for bnxt_re compatibility
export UCX_IB_TLS="${UCX_IB_TLS:-rc_verbs}"
export LMCACHE_USE_EXPERIMENTAL=True
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=0

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --api-key "$API_KEY" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --max-model-len "$MAX_MODEL_LEN" \
    --host 0.0.0.0 --port "$PORT" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    --no-enable-log-requests
