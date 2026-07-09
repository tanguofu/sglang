# GLM-5.2-FP8 SGLang Worker for AMD MI308X (gfx942)
# Based on official mi30x ROCm image — no patches, ENV-only adjustment.
#
# Build (from repo root):
#   docker build -f docker/rocm-mi308x-glm52/Dockerfile \
#     -t sglang-glm52-308x:latest .
#
# Run:
#   docker run -d --name sglang_308x \
#     --privileged --network host --shm-size 32g \
#     --device /dev/kfd --device /dev/dri --group-add video \
#     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
#     -v /data:/data \
#     sglang-glm52-308x:latest
#
# No code patches — relies on official sglang v0.5.14 HIP support.
# Environment variables are pre-set via ENV.
# The entrypoint (start_server.sh) only needs MODEL_PATH, API_KEY, PORT.

FROM lmsysorg/sglang-rocm:v0.5.14-rocm720-mi30x-20260708

LABEL maintainer="guofutan"
LABEL description="GLM-5.2-FP8 SGLang worker for AMD MI308X (gfx942) — no patches, official image + ENV"
LABEL sglang-base-image="v0.5.14-rocm720-mi30x-20260708"
LABEL gpu-arch="gfx942"

# ============================================================
# Environment Variables (21, based on MI355X production config)
# Key change: PYTORCH_ROCM_ARCH=gfx942 (was gfx950 for MI355X)
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
