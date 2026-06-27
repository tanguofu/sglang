#!/bin/bash
set -e

IMAGE="lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260623"
CONTAINER_NAME="sglang_dp8_ep8_mtp"

docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

docker run -d \
  --name ${CONTAINER_NAME} \
  --privileged \
  --network host \
  --shm-size 64g \
  --ipc=host \
  --pid=host \
  --ulimit memlock=-1 \
  --cap-add=IPC_LOCK \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /data:/data \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e HSA_ENABLE_SDMA=0 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e SGLANG_USE_AITER=1 \
  -e SGLANG_USE_ROCM700A=1 \
  -e SGLANG_DISABLE_CUDNN_CHECK=1 \
  -e SGLANG_ROCM_FUSED_DECODE_MLA=1 \
  -e SGLANG_SET_CPU_AFFINITY=1 \
  -e SGLANG_MOE_PADDING=1 \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT8 \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e MORI_SHMEM_HEAP_SIZE=8GB \
  -e NCCL_DEBUG=WARN \
  ${IMAGE} \
  bash -c "
    python3 /data/patch_glm_config.py 2>/dev/null || true
    python3 /data/patch_eplb.py 2>/dev/null || true
    exec python3 -m sglang.launch_server \
      --model-path /data/models/GLM-5.2-FP8 \
      --tp-size 8 \
      --dp-size 8 \
      --enable-dp-attention \
      --moe-a2a-backend mori \
      --pp-size 1 \
      --trust-remote-code \
      --host 0.0.0.0 \
      --port 30000 \
      --context-length 1048576 \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --mem-fraction-static 0.85 \
      --enable-aiter-allreduce-fusion \
      --enable-fused-qk-norm-rope \
      --schedule-conservativeness 0.5 \
      --chunked-prefill-size 4096 \
      --kv-cache-dtype fp8_e4m3 \
      --speculative-algorithm NEXTN \
      --speculative-num-steps 2 \
      --speculative-num-draft-tokens 3 \
      --speculative-eagle-topk 1 \
      --max-running-requests 128 \
      --watchdog-timeout 3600 \
      --log-level info
  "

echo "Container ${CONTAINER_NAME} started"
