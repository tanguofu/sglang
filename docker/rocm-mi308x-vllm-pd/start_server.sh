#!/usr/bin/env bash
# vLLM PD for GLM-5.2 on MI308X — optimized config based on reference
# Connector: NixlPush(P) / NixlPull(D) + LMCache local CPU on P
# All AITER + N-gram + UCX optimizations from reference config

set -euo pipefail

PD_ROLE="${PD_ROLE:-prefill}"
MODEL_PATH="${MODEL_PATH:-/data/model/glm52-fp8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.2}"
API_KEY="${API_KEY:-sk-46faecc9d0bc4dcd9db6a15c73ae91c8}"
PORT="${PORT:-8000}"
PEER_IP="${PEER_IP:-127.0.0.1}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-202752}"

echo "============================================"
echo " vLLM PD Worker (${PD_ROLE})"
echo "============================================"
echo " Model: $MODEL_PATH  MaxLen: $MAX_MODEL_LEN  Port: $PORT  TP: $TP_SIZE"

# --- Unset incompatible ---
unset PYTORCH_CUDA_ALLOC_CONF

# --- AITER + ROCm (full set from reference) ---
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_BLOCK_GEMM=1
export VLLM_ROCM_USE_AITER_FP8_BLOCK_MOE=1
export VLLM_ROCM_USE_AITER_ASMMOE=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1
export VLLM_AITER_TRITON_FUSED_ROPE_CACHE_CONCAT=1
export AITER_ENABLE_VSKIP=1

# --- vLLM V1 + MLA ---
export VLLM_USE_V1=1
export VLLM_ENABLE_MLA_QKV_MERGE=1
export VLLM_USE_TRTLLM_FUSEMOE=0
export VLLM_USE_GROUPED_TOPK_KERNEL=0
export VLLM_USE_TRITON_FLASH_ATTN=0

# --- Sparse attention (GLM-5.2 DSA) ---
export VLLM_SPARSE_ATTENTION=1
export VLLM_FUSED_SPARSE_MLA=1
export VLLM_FUSED_SPARSE_MLA_TP=1
export VLLM_SPARSE_K_CACHE_PADDING_SIZE=0

# --- MTP ---
export VLLM_FUSED_MTP_MODEL=1
export VLLM_MTP_REJECT_SAMPLE_METHOD="strict"
export VLLM_CPU_GPU_OVERLAP=1

# --- N-gram stopping (prevent reasoning repeat) ---
export VLLM_NGRAM_STOPPING_ENABLED=True
export VLLM_NGRAM_STOPPING_LENGTH=16
export VLLM_NGRAM_STOPPING_PATIENCE=4
export VLLM_TOPP_RESET_LENGTH=2
export VLLM_TOPP_RESET_VALUE=0.01
export VLLM_FORBID_REPEAT_THINK=1
export VLLM_FORBID_NO_THINK=1

# --- NIXL/UCX (from reference) ---
export NIXL_BACKEND=UCX
export VLLM_NIXL_ENABLE_FULL_TRANSPORT=1
export UCX_NET_DEVICES=$(ls /sys/class/infiniband 2>/dev/null | sed -z 's/\n/:1,/g' | sed 's/,$//')
export UCX_MAX_RMA_RAILS=2
export UCX_IB_TRAFFIC_CLASS=160
export UCX_IB_ROCE_REACHABILITY_MODE=all
export UCX_TLS=rc,rocm

# --- Misc ---
export VLLM_RPC_TIMEOUT=18000
export SAFETENSORS_FAST_GPU=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=36000
export PYTHONHASHSEED=0
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- Patch AITER torch.compile bug ---
SPARSE_INDEXER="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py"
if [ -f "$SPARSE_INDEXER" ] && grep -q "Sparse attention indexer ROCm path" "$SPARSE_INDEXER" ]; then
    echo "Patching AITER torch.compile..."
    python3 -c "
p = '$SPARSE_INDEXER'
with open(p) as f: c = f.read()
old = 'raise RuntimeError(\n            \"Sparse attention indexer ROCm path is only supported on AITER. \"\n            \"Please enable aiter with VLLM_ROCM_USE_AITER=1\"\n        )'
if old in c:
    c = c.replace(old, 'pass  # PATCHED')
    with open(p, 'w') as f: f.write(c)
    print('AITER patch applied')
"
fi

# --- LMCache (P self-cache) ---
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=360
export LMCACHE_RESERVE_LOCAL_CPU_SIZE=128
unset LMCACHE_REMOTE_URL 2>/dev/null || true
unset LMCACHE_ENABLE_PD 2>/dev/null || true

# --- Role-specific ---
if [ "$PD_ROLE" = "prefill" ]; then
    export VLLM_CPU_GPU_OVERLAP=0
    export VLLM_ENABLE_MLA_PURE_SP=1
    export VLLM_ENABLE_MLA_PURE_SPLB=1
    export VLLM_PREFILL_TOPK_OPT=1

    EXTRA_ARGS="--max-num-seqs 1 --max-num-batched-tokens 8192 --enforce-eager --gpu-memory-utilization ${GPU_MEM_UTIL}"

    KV_TRANSFER_CONFIG=$(cat <<EOF
{
    "kv_connector": "NixlPushConnector",
    "kv_role": "kv_producer",
    "kv_buffer_device": "cpu",
    "kv_buffer_size": 1000000000,
    "kv_ip": "${PEER_IP}",
    "kv_port": 14579,
    "kv_connector_extra_config": {"backends": ["UCX"]}
}
EOF
)

elif [ "$PD_ROLE" = "decode" ]; then
    export VLLM_MLA_FUSED_NORM=1
    export VLLM_MLA_FUSED_ROTARY=1
    export VLLM_DECODE_TOPK_OPT=1
    export VLLM_USE_NON_PERSISTENT_MLA=1

    EXTRA_ARGS="--max-num-seqs 6 --max-num-batched-tokens 192 --capture-scale 4 --capture-sizes 1 2 3 4 5 6 --gpu-memory-utilization 0.9"

    KV_TRANSFER_CONFIG=$(cat <<EOF
{
    "kv_connector": "NixlPullConnector",
    "kv_role": "kv_consumer",
    "kv_buffer_device": "cpu",
    "kv_buffer_size": 1000000000,
    "kv_ip": "${PEER_IP}",
    "kv_port": 14579,
    "kv_connector_extra_config": {"backends": ["UCX"]}
}
EOF
)
else
    echo "ERROR: Unknown PD_ROLE=$PD_ROLE"; exit 1
fi

echo " Extra: $EXTRA_ARGS"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --api-key "$API_KEY" \
    --tensor-parallel-size "$TP_SIZE" \
    --trust-remote-code \
    --no-enable-log-requests \
    --no-enable-prefix-caching \
    --max-model-len "$MAX_MODEL_LEN" \
    --block-size 64 \
    --distributed-executor-backend mp \
    --enable-reasoning \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --chat-template-content-format=string \
    --enable-prompt-tokens-details \
    --speculative-config '{"num_speculative_tokens":3, "method":"deepseek_mtp"}' \
    --compilation-config '{"custom_ops": ["+rms_norm"]}' \
    --system-prompt-num 2 \
    --host 0.0.0.0 --port "$PORT" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    $EXTRA_ARGS
