# GLM-5.2-FP8 SGLang Worker for AMD MI355X (gfx950)
# Master snapshot — exact reproduction of amd-355-master (216.128.153.58) as of 2026-07-09.
#
# This Dockerfile bakes all runtime patches into the image at build time.
# The result is functionally identical to the running master container, which
# used the 0706 base image + runtime patching via patch_0706_unified.py.
#
# Key differences from docker/rocm-mi355x-glm52-0708/:
#   - AITER stays at base image version (9127c94a1, 2026-06-25) — NOT upgraded
#   - Includes 06d: disable_dsa_fused_store (long-context correctness guard)
#   - Env vars match master exactly (including NCCL_DEBUG=INFO, NCCL_MIN_NCHANNELS=112)
#   - Launch command matches master exactly (steps=3, draft_tokens=4, mrr=32)
#
# Build (from repo root):
#   docker build -f docker/rocm-mi355x-glm52-master-snapshot/Dockerfile \
#     -t sglang-glm52-master-snapshot:latest .
#
# Run:
#   docker run -d --name sglang_master_snapshot \
#     --privileged --network host --shm-size 32g \
#     --device /dev/kfd --device /dev/dri --group-add video \
#     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
#     -v /data:/data \
#     sglang-glm52-master-snapshot:latest

FROM lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260706

LABEL maintainer="guofutan"
LABEL description="GLM-5.2-FP8 SGLang worker — exact snapshot of amd-355-master (2026-07-09)"
LABEL sglang-base-image="v0.5.14-rocm720-mi35x-20260706"
LABEL branch="amd355-master-snapshot"
LABEL aiter-commit="9127c94a18e4398e1eba91f6639e910f0994ad02"
LABEL source-machine="216.128.153.58 (amd-355-master)"
LABEL snapshot-date="2026-07-09"

# ============================================================
# Step 1: Copy patch scripts into /data/ (matching master layout)
# ============================================================
COPY docker/rocm-mi355x-glm52-master-snapshot/patches/ /data/

# ============================================================
# Step 2: Apply all patches at build time
# ============================================================
# This replicates exactly what the master container does at runtime:
#   python3 /data/patch_0706_unified.py && python3 /data/disable_dsa_fused_store.py
#
# patch_0706_unified.py applies:
#   01-05: HIP/DSA/MTP/AITER enablement (calls patch_sglang_glm52_rocm_all.py)
#   06a:   Supplement v4 (view->reshape, cos_sin_cache, dual_stream threshold)
#   06b:   D2H sync elimination (dsa_backend + frozen_kv_mtp_worker)
#   06c:   Draft extend CUDA graph enablement for HIP
#   06d:   Disable DSA fused-store on ROCm (long-context correctness guard)
#   Gen1:  AITER BF16 tuned config generation
#   Gen2:  A8W8 blockscale tuned config generation
#
# disable_dsa_fused_store.py is redundant with 06d (both make
# can_use_dsa_fused_store return False) but is included for faithfulness.
RUN python3 /data/patch_0706_unified.py && \
    python3 /data/disable_dsa_fused_store.py

# ============================================================
# Step 3: Set all environment variables (exact match to master)
# ============================================================
# These match the -e flags from start_worker_0706_no_fused.sh
# and the inherited base image env vars.
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV NCCL_DEBUG=INFO
ENV HSA_ENABLE_SDMA=0
ENV HIP_FORCE_DEV_KERNARG=1
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV NCCL_CUMEM_ENABLE=0
ENV NCCL_MIN_NCHANNELS=112
ENV NCCL_NVLS_ENABLE=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYTORCH_ROCM_ARCH=gfx950
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

# ============================================================
# Step 4: Install startup script as entrypoint
# ============================================================
COPY docker/rocm-mi355x-glm52-master-snapshot/start_server.sh /start_server.sh
RUN chmod +x /start_server.sh

ENTRYPOINT ["/start_server.sh"]
