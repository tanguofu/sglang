#!/usr/bin/env bash
# Start GLM-5.2-FP8 SGLang worker with MoriEP (DP=8 + EP=8 + low_latency)
set -euo pipefail

IMAGE=${IMAGE:-lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260706}
CONTAINER_NAME=${CONTAINER_NAME:-sglang_0706_ep_worker}
PORT=${PORT:-30000}
API_KEY=${API_KEY:-sk-46faecc9d0bc4dcd9db6a15c73ae91c8}
MODEL_PATH=${MODEL_PATH:-/data/models/GLM-5.2-FP8/}

PATCH_BUNDLE=${PATCH_BUNDLE:-/data/patch_sglang_glm52_rocm_all.py}
GEN_AITER=${GEN_AITER:-/data/gen_aiter_dense_0702_v2.py}
GEN_A8W8=${GEN_A8W8:-/data/gen_a8w8_dense.py}

for f in "$PATCH_BUNDLE" "$GEN_AITER" "$GEN_A8W8"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required file not found: $f" >&2
    exit 1
  fi
done

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  if [[ "${REPLACE:-0}" == "1" ]]; then
    echo "REPLACE=1: removing existing container $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME"
  else
    echo "ERROR: container $CONTAINER_NAME already exists. Set REPLACE=1 to remove." >&2
    exit 1
  fi
fi

if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  echo "ERROR: port $PORT is already listening" >&2
  exit 1
fi

echo "Starting $CONTAINER_NAME from $IMAGE on port $PORT"
echo "Config: DP=8 + EP=8 + MoriEP low_latency + FP8 dispatch"

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
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NCCL_DEBUG=INFO \
  -e HSA_ENABLE_SDMA=1 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_MIN_NCHANNELS=112 \
  -e NCCL_NVLS_ENABLE=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT8 \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e SGLANG_DISABLE_CUDNN_CHECK=1 \
  -e SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM=1 \
  -e SGLANG_INT4_WEIGHT=0 \
  -e SGLANG_MOE_PADDING=1 \
  -e SGLANG_ROCM_DISABLE_LINEARQUANT=0 \
  -e SGLANG_ROCM_FUSED_DECODE_MLA=1 \
  -e SGLANG_SET_CPU_AFFINITY=1 \
  -e SGLANG_USE_AITER=1 \
  -e SGLANG_USE_ROCM700A=1 \
  -e SGLANG_MORI_DISPATCH_DTYPE=fp8 \
  -e SGLANG_MORI_COMBINE_DTYPE=bf16 \
  -e MORI_ENABLE_SDMA=1 \
  -e MORI_SHMEM_HEAP_SIZE=17179869184 \
  -e SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=4096 \
  -v /data:/data \
  "$IMAGE" \
  bash -c "
    python3 $PATCH_BUNDLE && \
    python3 /data/patch_0706_supplement_v4.py && \
    python3 $GEN_AITER && \
    python3 $GEN_A8W8 && \
    exec python3 -m sglang.launch_server \
      --model-path $MODEL_PATH \
      --model-impl sglang \
      --served-model-name glm-5.2 \
      --api-key $API_KEY \
      --tp-size 8 --pp-size 1 --trust-remote-code \
      --host 0.0.0.0 --port $PORT \
      --context-length 1048576 \
      --tool-call-parser glm47 --reasoning-parser glm45 \
      --mem-fraction-static 0.88 \
      --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 12 16 \
      --cuda-graph-max-bs-decode 16 \
      --chunked-prefill-size 32768 \
      --enable-fused-qk-norm-rope \
      --schedule-conservativeness 0.5 \
      --prefill-max-requests 32 --max-prefill-tokens 32768 \
      --kv-cache-dtype fp8_e4m3 \
      --speculative-algorithm NEXTN \
      --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
      --speculative-eagle-topk 1 \
      --cuda-graph-backend-prefill breakable \
      --max-running-requests 32 \
      --cuda-graph-bs-prefill 4 8 16 32 \
      --enable-metrics --skip-server-warmup \
      --dp-size 8 --ep-size 8 \
      --moe-a2a-backend mori \
      --deepep-mode low_latency \
      --enable-dp-attention \
      --enable-p2p-check \
      --watchdog-timeout 3600 --log-level info
  "

echo "Started. Check logs with: docker logs -f $CONTAINER_NAME"
