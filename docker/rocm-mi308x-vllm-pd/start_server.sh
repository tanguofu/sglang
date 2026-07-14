#!/usr/bin/env bash
# Entrypoint for vLLM PD worker with NixlConnector UCX RDMA
# Branch: 308x-vllm-llcache-1pd
# Prefill: NixlPushConnector | Decode: NixlPullConnector
# kv_buffer_device=cpu → host memory RDMA (no peer_mem needed)
# AITER torch.compile patch at startup → enables CUDA graph

set -euo pipefail

PD_ROLE="${PD_ROLE:-prefill}"
MODEL_PATH="${MODEL_PATH:-/data/model/glm52-fp8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.2}"
API_KEY="${API_KEY:-sk-46faecc9d0bc4dcd9db6a15c73ae91c8}"
PORT="${PORT:-13000}"
HOST_IP="${HOST_IP:-0.0.0.0}"
PEER_IP="${PEER_IP:-127.0.0.1}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"

echo "============================================"
echo " vLLM PD Worker (${PD_ROLE})"
echo "============================================"
echo " Model: $MODEL_PATH  MaxLen: $MAX_MODEL_LEN  GPUMem: $GPU_MEM_UTIL"
echo " Port: $PORT  TP: $TP_SIZE  Host: $HOST_IP  Peer: $PEER_IP"

unset PYTORCH_CUDA_ALLOC_CONF
export VLLM_ROCM_USE_AITER=1

# Patch AITER torch.compile bug — remove raise RuntimeError that breaks
# symbolic tracing (runtime check still works, just no crash during compile)
SPARSE_INDEXER="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py"
if [ -f "$SPARSE_INDEXER" ] && grep -q "Sparse attention indexer ROCm path" "$SPARSE_INDEXER"; then
    echo "Patching AITER torch.compile check..."
    sed -i 's/raise RuntimeError(/# PATCHED: raise RuntimeError(/' "$SPARSE_INDEXER"
    echo "AITER patch applied → CUDA graph can be enabled"
fi

# PD connector: prefill=push, decode=pull
if [ "$PD_ROLE" = "prefill" ]; then
    KV_CONNECTOR="NixlPushConnector"
    KV_ROLE="kv_producer"
elif [ "$PD_ROLE" = "decode" ]; then
    KV_CONNECTOR="NixlPullConnector"
    KV_ROLE="kv_consumer"
else
    echo "ERROR: Unknown PD_ROLE=$PD_ROLE"; exit 1
fi

KV_TRANSFER_CONFIG=$(cat <<EOF
{
    "kv_connector": "${KV_CONNECTOR}",
    "kv_role": "${KV_ROLE}",
    "kv_buffer_device": "cpu",
    "kv_buffer_size": 1000000000,
    "kv_ip": "${PEER_IP}",
    "kv_port": 14579,
    "kv_connector_extra_config": {"backends": ["UCX"]}
}
EOF
)

export UCX_TLS="${UCX_TLS:-rc_verbs,tcp,cuda_copy,self}"
unset UCX_NET_DEVICES
export UCX_IB_TLS="${UCX_IB_TLS:-rc_verbs}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-bnxt_re}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_DEBUG=INFO
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
