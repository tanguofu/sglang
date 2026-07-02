#!/bin/bash
# Launch GLM-5.2 for PRECISION EVALUATION — no speculative decoding.
#
# This is the cleanest baseline for precision validation:
#   1. --kv-cache-dtype auto  (→ bf16 on gfx950, NOT fp8_e4m3)
#   2. --context-length 131072 (128K, enough for 64K max_tokens + input)
#   3. --reasoning-parser glm45 + --tool-call-parser glm47
#   4. NO speculative decoding (eliminates DSpark crash risk)
#   5. CUDA graph enabled (for speed; does not affect precision in greedy)
#
# Usage: bash start_precision_eval.sh [port]

set -e

IMAGE="lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260626"
PORT=${1:-30000}

docker rm -f glm52_precision_eval 2>/dev/null || true
sleep 2

docker run -d --name glm52_precision_eval \
  --device /dev/kfd --device /dev/dri --network host --shm-size 32G --ipc host \
  -v /data:/data \
  -e SGLANG_USE_AITER=1 -e SGLANG_DISABLE_CUDNN_CHECK=1 \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 -e SGLANG_INT4_WEIGHT=0 \
  -e SGLANG_MOE_PADDING=1 -e SGLANG_ROCM_FUSED_DECODE_MLA=1 \
  -e SGLANG_SET_CPU_AFFINITY=1 -e SGLANG_USE_ROCM700A=1 \
  -e SGLANG_ROCM_DISABLE_LINEARQUANT=0 \
  -e NCCL_SOCKET_IFNAME=enp193s0f0np0 \
  -e NCCL_MIN_NCHANNELS=112 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e "PYTORCH_ROCM_ARCH=gfx942;gfx950" \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT8 \
  "$IMAGE" \
  bash -c "python3 /data/patch_glm_config.py 2>/dev/null || true && \
    python3 /data/patch_dsa_backend_v2.py && \
    python3 /data/gen_aiter_dense.py && \
    python3 /data/gen_a8w8_dense.py && \
    exec python3 -m sglang.launch_server \
      --model-path /data/models/GLM-5.2-FP8 \
      --tp-size 8 --pp-size 1 --trust-remote-code \
      --host 0.0.0.0 --port $PORT \
      --context-length 1048576 \
      --mem-fraction-static 0.90 \
      --enable-aiter-allreduce-fusion \
      --enable-mixed-chunk \
      --chunked-prefill-size 32768 \
      --enable-fused-qk-norm-rope \
      --schedule-conservativeness 0.5 \
      --prefill-max-requests 32 \
      --max-prefill-tokens 32768 \
      --kv-cache-dtype auto \
      --max-running-requests 128 \
      --reasoning-parser glm45 \
      --tool-call-parser glm47 \
      --watchdog-timeout 3600 --log-level info"

echo "Precision eval server starting on port $PORT (no speculative decoding)"
echo "Check logs: docker logs -f glm52_precision_eval"
