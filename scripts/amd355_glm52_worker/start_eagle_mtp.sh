#!/bin/bash
# Launch GLM-5.2 with EAGLE MTP speculative decoding — official GLM-5.2 method.
#
# This is the official speculative decoding path for GLM-5.2 (uses the model's
# built-in nextn MTP layer, NOT an external draft model). Verified compatible
# with DSA attention backend.
#
# Key config:
#   1. --kv-cache-dtype auto  (→ bf16 on gfx950, precision-safe)
#   2. --context-length 1048576 (1M, production target)
#   3. --speculative-algorithm EAGLE (NOT NEXTN — EAGLE is the registered name)
#   4. --speculative-num-steps 2 --speculative-num-draft-tokens 3 --speculative-eagle-topk 1
#      (verified 3.46-3.68x speedup, accept_len ~2.85)
#   5. --reasoning-parser glm45 + --tool-call-parser glm47
#   6. CUDA graph enabled (for speed; greedy verify is precision-safe)
#
# Usage: bash start_eagle_mtp.sh [port]

set -e

IMAGE="lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260626"
PORT=${1:-30000}

docker rm -f glm52_eagle_mtp 2>/dev/null || true
sleep 2

docker run -d --name glm52_eagle_mtp \
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
      --mem-fraction-static 0.88 \
      --enable-aiter-allreduce-fusion \
      --enable-mixed-chunk \
      --chunked-prefill-size 32768 \
      --enable-fused-qk-norm-rope \
      --schedule-conservativeness 0.5 \
      --prefill-max-requests 32 \
      --max-prefill-tokens 32768 \
      --kv-cache-dtype auto \
      --max-running-requests 128 \
      --speculative-algorithm EAGLE \
      --speculative-num-steps 2 \
      --speculative-num-draft-tokens 3 \
      --speculative-eagle-topk 1 \
      --reasoning-parser glm45 \
      --tool-call-parser glm47 \
      --watchdog-timeout 3600 --log-level info"

echo "EAGLE MTP server starting on port $PORT (1M context, BF16 KV, CUDA graph on)"
echo "Check logs: docker logs -f glm52_eagle_mtp"
