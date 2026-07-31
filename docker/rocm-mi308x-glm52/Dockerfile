# GLM-5.2-FP8 SGLang Worker for AMD MI308X (gfx942)
# Based on official mi30x ROCm image + upstream/main + PR #31047 + #31260 (GLM-5.2 DSpark)
#
# Build (from repo root):
#   docker build -f docker/rocm-mi308x-glm52/Dockerfile #     -t sglang-glm52-dspark-308x:latest .
#
# Run:
#   docker run -d --name sglang_308x #     --privileged --network host --shm-size 32g #     --device /dev/kfd --device /dev/dri --group-add video #     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined #     -v /data:/data #     sglang-glm52-dspark-308x:latest

FROM lmsysorg/sglang-rocm:v0.5.15.post1-rocm720-mi30x-20260718

LABEL maintainer="guofutan"
LABEL description="GLM-5.2-FP8 SGLang worker for AMD MI308X (gfx942) with DSpark (#31047+#31260)"
LABEL sglang-base-image="v0.5.15.post1-rocm720-mi30x-20260718"
LABEL gpu-arch="gfx942"
LABEL branch="feat/glm52-dspark-308x-official"

# ============================================================
# Step 1: Replace sglang source with upstream/main + #31047 + #31260
# ============================================================
# The official image has sglang installed in editable mode pointing to
# /sgl-workspace/sglang/python/sglang/. We overwrite with our branch
# (upstream/main + GLM-5.2 DSpark PRs).
COPY python/sglang/ /sgl-workspace/sglang/python/sglang/

# Verify DSpark PRs are present (fail build if missing)
RUN python3 -c "\
src = open('/sgl-workspace/sglang/python/sglang/srt/models/dspark.py').read(); \
assert 'DSparkDraftModel' in src, 'PR #31260 MISSING: DSparkDraftModel entry class'; \
assert 'Qwen3DSparkModel' in src, 'PR MISSING: Qwen3DSparkModel'; \
src2 = open('/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py').read(); \
assert 'not _is_hip' in src2, 'PR #31260 MISSING: topk v2 disabled on HIP'; \
src3 = open('/sgl-workspace/sglang/python/sglang/srt/utils/hf_transformers/config.py').read(); \
assert '_try_load_dspark_draft_config' in src3, 'PR #31047 MISSING: hf config loader'; \
src4 = open('/sgl-workspace/sglang/python/sglang/srt/speculative/dspark_components/dspark_config.py').read(); \
assert '_get_speculators_config' in src4, 'PR #31047 MISSING: speculators_config parser'; \
print('DSpark PRs #31047 + #31260 verified OK')"

# ============================================================
# Step 1b: Add FlyDSL gfx942 fp8 MQA logits kernel + patch Triton path
# MI308X (gfx942) has 64KB shared memory limit vs MI355X's 80KB+.
# ============================================================
COPY docker/rocm-mi308x-glm52/patches/flydsl/fp8_mqa_logits.py /sgl-workspace/aiter/aiter/ops/flydsl/kernels/fp8_mqa_logits.py
COPY docker/rocm-mi308x-glm52/patches/flydsl/__init__.py /sgl-workspace/aiter/aiter/ops/flydsl/__init__.py
COPY docker/rocm-mi308x-glm52/patches/fp8_mqa_logits.py /sgl-workspace/aiter/aiter/ops/triton/attention/fp8_mqa_logits.py

# ============================================================
# Step 2: Environment Variables (gfx942 specific)
# ============================================================
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV NCCL_DEBUG=INFO
ENV HSA_ENABLE_SDMA=0
ENV HIP_FORCE_DEV_KERNARG=1
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV NCCL_CUMEM_ENABLE=0
ENV NCCL_MIN_NCHANNELS=112
ENV NCCL_NVLS_ENABLE=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYTORCH_ROCM_ARCH=gfx942
ENV ROCM_QUICK_REDUCE_QUANTIZATION=INT8
ENV SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
ENV SGLANG_DISABLE_CUDNN_CHECK=1
ENV SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM=1
ENV SGLANG_INT4_WEIGHT=0
ENV SGLANG_MOE_PADDING=1
ENV SGLANG_ROCM_DISABLE_LINEARQUANT=0
ENV SGLANG_ROCM_FUSED_DECODE_MLA=1
ENV SGLANG_SET_CPU_AFFINITY=1
ENV SGLANG_USE_AITER=1
ENV SGLANG_USE_ROCM700A=1

# Runtime-overridable params
ENV MODEL_PATH=/data/model/glm52-fp8
ENV PORT=30000
ENV API_KEY=sk-46faecc9d0bc4dcd9db6a15c73ae91c8

# ============================================================
# Entrypoint
# ============================================================
COPY docker/rocm-mi308x-glm52/start_server.sh /start_server.sh
RUN chmod +x /start_server.sh

ENTRYPOINT ["/start_server.sh"]
