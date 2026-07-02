#!/bin/bash
# Launch GLM-5.2 with DSpark speculative decoding (Route B, PR #29538)
#
# Based on sglang PR #29538 (independent DSpark worker with CUDA graph support)
# + is_hip() fix for AMD ROCm
# + Glm5ForCausalLMDSpark model (adapted from DeepseekV4ForCausalLMDSpark)
#
# Key design:
#   - CUDA graph ENABLED (is_hip() fix in init_cuda_graphs)
#   - NCCL_P2P_DISABLE=1 (draft worker NCCL IPC workaround)
#   - NO --enable-aiter-allreduce-fusion (incompatible with draft worker)
#   - kv-cache-dtype auto (→ bfloat16 on gfx950)
#   - Requires Glm5ForCausalLMDSpark checkpoint (NOT DFlashDraftModel)
#
# Usage: bash start_dspark.sh [image_version] [port]

IMAGE=${1:-glm52-dspark:v0.5.25}
PORT=${2:-30000}

docker rm -f glm52_dspark 2>/dev/null
sleep 2

docker run -d --name glm52_dspark \
  --device /dev/kfd --device /dev/dri --network host --shm-size 32G --ipc host \
  --privileged \
  -v /data:/data \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e SGLANG_USE_AITER=1 -e SGLANG_DISABLE_CUDNN_CHECK=1 \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 -e SGLANG_INT4_WEIGHT=0 \
  -e SGLANG_MOE_PADDING=1 -e SGLANG_ROCM_FUSED_DECODE_MLA=1 \
  -e SGLANG_SET_CPU_AFFINITY=1 -e SGLANG_USE_ROCM700A=1 \
  -e SGLANG_ROCM_DISABLE_LINEARQUANT=0 \
  -e NCCL_SOCKET_IFNAME=enp193s0f0np0 \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_DEBUG=WARN \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e "PYTORCH_ROCM_ARCH=gfx942;gfx950" \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT8 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HSA_ENABLE_SDMA=0 \
  "$IMAGE" \
  bash -c "python3 /data/patch_glm_config.py 2>/dev/null || true && \
    python3 /data/patch_dsa_backend_v2.py 2>/dev/null || true && \
    python3 /data/gen_aiter_dense.py 2>/dev/null || true && \
    python3 /data/gen_a8w8_dense.py 2>/dev/null || true && \
    cp /data/dspark_v3_files/dspark_worker_v2.py /sgl-workspace/sglang/python/sglang/srt/speculative/dspark_worker_v2.py && \
    cp /data/dspark_v3_files/dspark_info.py /sgl-workspace/sglang/python/sglang/srt/speculative/dspark_info.py && \
    cp /data/dspark_v3_files/glm5_dspark.py /sgl-workspace/sglang/python/sglang/srt/models/glm5_dspark.py && \
    cp /data/dspark_v3_files/deepseek_v2.py /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py && \
    cp /data/dspark_v3_files/dflash_worker_v2.py /sgl-workspace/sglang/python/sglang/srt/speculative/dflash_worker_v2.py && \
    cp /data/dspark_v3_files/spec_info.py /sgl-workspace/sglang/python/sglang/srt/speculative/spec_info.py && \
    cp /data/dspark_v3_files/spec_registry.py /sgl-workspace/sglang/python/sglang/srt/speculative/spec_registry.py && \
    cp /data/dspark_v3_files/spec_utils.py /sgl-workspace/sglang/python/sglang/srt/speculative/spec_utils.py && \
    cp /data/dspark_v3_files/speculative_hook.py /sgl-workspace/sglang/python/sglang/srt/arg_groups/speculative_hook.py && \
    cp /data/dspark_v3_files/server_args.py /sgl-workspace/sglang/python/sglang/srt/server_args.py && \
    cp /data/dspark_v3_files/model_runner.py /sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py && \
    cp /data/dspark_v3_files/decode_cuda_graph_runner.py /sgl-workspace/sglang/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py && \
    exec python3 -m sglang.launch_server \
      --model-path /data/models/GLM-5.2-FP8 \
      --tp-size 8 --pp-size 1 --trust-remote-code \
      --host 0.0.0.0 --port $PORT \
      --context-length 1048576 --mem-fraction-static 0.82 \
      --enable-fused-qk-norm-rope \
      --chunked-prefill-size 32768 --schedule-conservativeness 0.5 \
      --prefill-max-requests 32 --max-prefill-tokens 32768 \
      --kv-cache-dtype auto \
      --max-running-requests 128 \
      --speculative-algorithm DSPARK \
      --speculative-draft-model-path /data/checkpoints/deepspec/dspark_glm5_mock/step_latest \
      --speculative-num-steps 1 --speculative-num-draft-tokens 7 \
      --speculative-eagle-topk 1 \
      --reasoning-parser glm45 --tool-call-parser glm47 \
      --weight-loader-disable-mmap \
      --watchdog-timeout 3600 --log-level info"

echo "DSpark server starting on port $PORT with image $IMAGE"
echo "Route B (PR #29538 + is_hip fix + Glm5ForCausalLMDSpark)"
echo "Check logs: docker logs -f glm52_dspark"
