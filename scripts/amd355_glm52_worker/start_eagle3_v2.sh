#!/bin/bash
set -e
IMAGE="lmsysorg/sglang-rocm:v0.5.13.post1-rocm720-mi35x-20260623"
CONTAINER_NAME="sglang_glm52_eagle3v2"
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true
docker run -d \
  --name ${CONTAINER_NAME} --restart no --privileged --network host --shm-size 32g \
  -v /data:/data --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e NCCL_DEBUG=INFO -e HSA_ENABLE_SDMA=0 \
  -e SGLANG_DSA_FUSE_TOPK=false \
  ${IMAGE} \
  bash -c "
    python3 /data/patch_glm_config.py 2>/dev/null || true
    exec python3 -m sglang.launch_server \
      --model-path /data/models/GLM-5.2-FP8 --tp-size 8 --pp-size 1 --trust-remote-code \
      --host 0.0.0.0 --port 30000 --context-length 1048576 \
      --tool-call-parser glm47 --reasoning-parser glm45 \
      --mem-fraction-static 0.88 \
      --enable-aiter-allreduce-fusion --enable-mixed-chunk --chunked-prefill-size 32768 \
      --enable-fused-qk-norm-rope --schedule-conservativeness 0.5 \
      --prefill-max-requests 128 --max-prefill-tokens 32768 --kv-cache-dtype fp8_e4m3 \
      --speculative-algorithm EAGLE3 --speculative-num-steps 5 --speculative-num-draft-tokens 6 \
      --speculative-eagle-topk 1 --page-size 1 \
      --max-running-requests 128 --watchdog-timeout 3600 --log-level info
  "
echo "Container ${CONTAINER_NAME} started"
