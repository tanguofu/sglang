#!/bin/bash
docker rm -f sglang_mtp3_nomultistream 2>/dev/null
docker run -d \
  --name sglang_mtp3_nomultistream \
  --privileged --network host --ipc host --shm-size 32g \
  --device /dev/kfd --device /dev/dri \
  -v /data:/data \
  -e SGLANG_SET_CPU_AFFINITY=1 -e NCCL_CUMEM_ENABLE=0 \
  -e SGLANG_USE_ROCM700A=1 -e HSA_ENABLE_SDMA=0 \
  -e GPU_ARCH_LIST=gfx950 -e SGLANG_INT4_WEIGHT=0 \
  -e SGLANG_USE_AITER=1 -e SGLANG_ROCM_FUSED_DECODE_MLA=1 \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  -e SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM=1 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e HSA_NO_SCRATCH_RECLAIM=1 -e SGLANG_MOE_PADDING=1 \
  -e SGLANG_ROCM_DISABLE_LINEARQUANT=0 -e NCCL_DEBUG=INFO \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e NCCL_NVLS_ENABLE=0 -e HIP_FORCE_DEV_KERNARG=1 \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT8 \
  -e SGLANG_DISABLE_CUDNN_CHECK=1 -e NCCL_MIN_NCHANNELS=112 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260702 \
  bash -c 'python3 /data/patch_glm_config.py 2>/dev/null || true &&\
    python3 /data/patch_dsa_backend_v2.py 2>/dev/null || true &&\
    python3 /data/patch_dsa_draft_extend.py && \
    python3 /data/patch_dsa_indexer_graph.py && \
    python3 /data/patch_disable_mha_swap.py && \
    python3 /data/patch_deterministic_argmax.py && \
    python3 /data/patch_draft_forward_argmax.py && \
    python3 /data/patch_hip_fusion_dual_stream_0702_v6.py && \
    python3 /data/patch_dual_stream_kw_fix.py && \
    python3 /data/patch_draft_alt_stream.py && \
    python3 /data/patch_fp8_view_fix.py && \
    python3 /data/patch_tp_style_0702.py && \
    python3 /data/patch_cuda_fp8_include.py && \
    python3 /data/gen_aiter_dense_0702_v2.py && \
    python3 /data/gen_a8w8_dense.py && \
    exec python3 -m sglang.launch_server \
      --model-path /data/models/GLM-5.2-FP8 \
      --model-impl sglang \
      --served-model-name glm-5.2 \
      --api-key sk-46faecc9d0bc4dcd9db6a15c73ae91c8 \
      --tp-size 8 --pp-size 1 --trust-remote-code \
      --host 0.0.0.0 --port 30000 \
      --context-length 1048576 \
      --tool-call-parser glm47 --reasoning-parser glm45 \
      --mem-fraction-static 0.88 \
      --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 12 16 \
      --cuda-graph-max-bs-decode 16 \
      --enable-aiter-allreduce-fusion --enable-mixed-chunk \
      --chunked-prefill-size 32768 \
      --enable-fused-qk-norm-rope \
      --schedule-conservativeness 0.5 \
      --prefill-max-requests 32 --max-prefill-tokens 32768 \
      --kv-cache-dtype fp8_e4m3 \
      --speculative-algorithm NEXTN \
      --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
      --speculative-eagle-topk 1 \
      --cuda-graph-backend-prefill breakable \
      --cuda-graph-bs-prefill 4 8 16 32 \
      --max-running-requests 32 \
      --enable-metrics --skip-server-warmup \
      --watchdog-timeout 3600 --log-level info'
