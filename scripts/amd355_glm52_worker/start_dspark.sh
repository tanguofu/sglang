#!/bin/bash
# Launch GLM-5.2 with DSpark speculative decoding on amd-355-worker (8x MI355X)
# Usage: bash start_dspark.sh [image_version] [port]
#
# Prerequisites:
#   - Docker image glm52-dspark:v0.5.17 (or newer)
#   - GLM-5.2-FP8 model at /data/models/GLM-5.2-FP8
#   - DSpark checkpoint at /data/dspark_checkpoints/deepspec/dspark_glm5_2/step_780
#   - Patch scripts at /data/patch_glm_config.py, /data/patch_dsa_backend_v2.py,
#     /data/gen_aiter_dense.py, /data/gen_a8w8_dense.py

IMAGE=${1:-glm52-dspark:v0.5.17}
PORT=${2:-30000}

docker rm -f glm52_dspark_test 2>/dev/null
sleep 2

docker run -d --name glm52_dspark_test \
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
      --context-length 8192 --mem-fraction-static 0.88 \
      --enable-aiter-allreduce-fusion --enable-mixed-chunk \
      --chunked-prefill-size 32768 --enable-fused-qk-norm-rope \
      --schedule-conservativeness 0.5 --prefill-max-requests 32 \
      --max-prefill-tokens 32768 --kv-cache-dtype fp8_e4m3 \
      --max-running-requests 128 \
      --speculative-algorithm DSPARK \
      --speculative-draft-model-path /data/dspark_checkpoints/deepspec/dspark_glm5_2/step_780 \
      --speculative-num-steps 2 --speculative-num-draft-tokens 7 \
      --speculative-eagle-topk 1 \
      --weight-loader-disable-mmap --disable-cuda-graph \
      --watchdog-timeout 3600 --log-level info"

echo "DSpark server starting on port $PORT with image $IMAGE"
echo "Check logs: docker logs -f glm52_dspark_test"
