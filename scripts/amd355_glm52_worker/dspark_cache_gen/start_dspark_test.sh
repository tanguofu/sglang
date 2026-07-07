#!/bin/bash
# Launch GLM-5.2 with DSpark speculative decoding for accept_rate testing
# Uses the same image + sglang source as cache gen, with DSPARK enabled
#
# Usage: bash start_dspark_test.sh <checkpoint_path> [port]

set -e

CHECKPOINT=${1:?Usage: bash $0 <checkpoint_path> [port]}
PORT=${2:-30000}

IMAGE="lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260704"
MODEL_PATH="/data/models/GLM-5.2-FP8"
CONTAINER_NAME="glm52_dspark_test"

echo "[$(date)] Launching DSpark test server"
echo "  checkpoint=$CHECKPOINT"
echo "  port=$PORT"

docker rm -f ${CONTAINER_NAME} 2>/dev/null || true
sleep 1

docker run -d --name ${CONTAINER_NAME} \
  --device /dev/kfd --device /dev/dri --network host --shm-size 32G --ipc host \
  --privileged -v /data:/data \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e SGLANG_USE_AITER=1 \
  -e SGLANG_DISABLE_CUDANN_CHECK=1 \
  -e SGLANG_MOE_PADDING=1 \
  -e SGLANG_ROCM_FUSED_DECODE_MLA=1 \
  -e SGLANG_SET_CPU_AFFINITY=1 \
  -e SGLANG_USE_ROCM700A=1 \
  -e NCCL_SOCKET_IFNAME=enp193s0f0np0 \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_DEBUG=WARN \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e PYTORCH_ROCM_ARCH="gfx942;gfx950" \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT8 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HSA_ENABLE_SDMA=0 \
  -e PYTHONPATH=/data/sglang_src/python \
  "$IMAGE" \
  bash -c "python3 /data/patch_glm_config.py 2>/dev/null || true; \
    python3 /data/patch_dsa_backend_v2.py 2>/dev/null || true; \
    exec python3 -m sglang.launch_server \
      --model-path ${MODEL_PATH} \
      --tp-size 8 --pp-size 1 --trust-remote-code \
      --host 0.0.0.0 --port ${PORT} \
      --context-length 4096 \
      --mem-fraction-static 0.82 \
      --enable-fused-qk-norm-rope \
      --chunked-prefill-size 32768 --schedule-conservativeness 0.5 \
      --prefill-max-requests 32 --max-prefill-tokens 32768 \
      --kv-cache-dtype auto \
      --max-running-requests 128 \
      --quantization fp8 \
      --speculative-algorithm DSPARK \
      --speculative-draft-model-path ${CHECKPOINT} \
      --speculative-draft-model-quantization unquant \
      --speculative-num-steps 1 --speculative-num-draft-tokens 7 \
      --speculative-eagle-topk 1 \
      --speculative-draft-attention-backend aiter \
      --disable-cuda-graph \
      --reasoning-parser glm45 --tool-call-parser glm47 \
      --watchdog-timeout 3600 --log-level info"

echo "[$(date)] DSpark server starting on port $PORT"
echo "  Container: ${CONTAINER_NAME}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Monitor: docker logs -f ${CONTAINER_NAME}"
echo "  Health: curl http://localhost:${PORT}/health"
echo "  Test: python3 /data/test_accept_rate.py http://localhost:${PORT}"
