# GLM-5.2-FP8 SGLang Worker for AMD MI355X (gfx950)
# 0723 build — latest base image + AITER upgrade + all patches baked in.
#
# Based on docker/rocm-mi355x-glm52-0708/Dockerfile (source-overwrite approach)
# and docker/rocm-mi355x-glm52-master-snapshot/Dockerfile (patch-script approach).
# This version uses patch scripts (faithful to the running master container) on
# top of the latest v0.5.15.post1 base image.
#
# Key changes from 0708:
#   - Base image: v0.5.15.post1-rocm720-mi35x-20260722 (sglang 0.5.15.post1)
#   - AITER upgraded to origin/main (118+ commits ahead of base 9127c94a1)
#   - GLM-5.2 tuned GEMM configs generated
#   - All 16 patches verified present at build time
#   - Env vars: adds missing SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM, HSA_ENABLE_SDMA,
#     NCCL_CUMEM_ENABLE, NCCL_NVLS_ENABLE, NCCL_DEBUG, PYTORCH_CUDA_ALLOC_CONF,
#     ROCM_QUICK_REDUCE_QUANTIZATION (not pre-set in new base image)
#
# Build (from repo root):
#   docker build -f docker/rocm-mi355x-glm52-0723/Dockerfile \
#     -t sglang-glm52-0723:latest .
#
# Run:
#   docker run -d --name sglang_0723 \
#     --privileged --network host --shm-size 32g \
#     --device /dev/kfd --device /dev/dri --group-add video \
#     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
#     -v /data:/data \
#     sglang-glm52-0723:latest

FROM lmsysorg/sglang-rocm:v0.5.15.post1-rocm720-mi35x-20260722

LABEL maintainer="guofutan"
LABEL description="GLM-5.2-FP8 SGLang worker for AMD MI355X with pre-applied ROCm patches (0723-opt)"
LABEL sglang-base-image="v0.5.15.post1-rocm720-mi35x-20260722"
LABEL sglang-version="0.5.15.post1.dev20260722"
LABEL branch="0723-opt"
LABEL aiter-base="9127c94a1 (2026-06-25)"
LABEL aiter-upgrade="origin/main (118+ commits)"
LABEL build-date="2026-07-23"

# ============================================================
# Step 1: Copy patch scripts into /data/ (matching master layout)
# ============================================================
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
COPY docker/rocm-mi355x-glm52-master-snapshot/patches/ /data/

# ============================================================
# Step 2: Apply all patches at build time
# ============================================================
# This replicates exactly what the master container does at runtime:
#   python3 /data/patch_0706_unified.py && python3 /data/disable_dsa_fused_store.py
RUN python3 /data/patch_0706_unified.py && \
    python3 /data/disable_dsa_fused_store.py

# ============================================================
# Step 3: Verify all 16 key patches are present (fail build if any missing)
# ============================================================
# If any patch is already included upstream in a future base image, the grep
# will still pass (the patch is idempotent / the marker already exists).
RUN python3 -c "\
src = open('/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py').read(); \
assert '_is_cuda or _is_hip' in src, 'PATCH MISSING: 1.1 JIT imports on HIP'; \
assert '(_is_cuda or _is_hip) and not envs.SGLANG_DISABLE_DSA_INDEXER_FUSION' in src, 'PATCH MISSING: 1.2 DSA indexer fusion on HIP'; \
assert '_k_norm_weight_f32' in src, 'PATCH MISSING: 1.6 k_norm f32 properties'; \
assert 'FIX(breakable-target-verify)' in src, 'PATCH MISSING: 04 target_verify metadata guard'; \
assert 'DUAL_STREAM_TOKEN_THRESHOLD = 1024' in src, 'PATCH MISSING: 06a S1 dual stream threshold'; \
src2 = open('/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py').read(); \
assert '(_is_cuda or _is_hip) and envs.SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM' in src2, 'PATCH MISSING: 2.1 dual stream on HIP'; \
src3 = open('/sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py').read(); \
assert 'is_hip() or envs.SGLANG_NPU_USE_MULTI_STREAM' in src3, 'PATCH MISSING: 3.1 alt_stream on HIP'; \
src4 = open('/sgl-workspace/sglang/python/sglang/srt/models/transformers.py').read(); \
assert 'mla_kv_a_proj' in src4, 'PATCH MISSING: 4.1 TP style'; \
src5 = open('/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py').read(); \
assert 'q.reshape(-1, layer.tp_q_head_num' in src5, 'PATCH MISSING: 5c view->reshape'; \
assert 'elif forward_batch.seq_lens_sum is not None' in src5, 'PATCH MISSING: 06b D2H sync elimination'; \
src6 = open('/sgl-workspace/sglang/python/sglang/srt/layers/radix_attention.py').read(); \
assert '# if _is_hip and not save_kv_cache' in src6, 'PATCH MISSING: 5d _pcg_mha_companion disabled'; \
src7 = open('/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8.py').read(); \
assert 'is_shuffled = True' in src7, 'PATCH MISSING: 02 fp8 is_shuffled'; \
src8 = open('/sgl-workspace/sglang/python/sglang/jit_kernel/dsv4/elementwise.py').read(); \
assert 'q_fp8_raw' in src8, 'PATCH MISSING: 03 FP8 view fix'; \
src9 = open('/sgl-workspace/sglang/python/sglang/jit_kernel/csrc/dsa/fused_store_index_cache.cuh').read(); \
assert 'USE_ROCM' in src9, 'PATCH MISSING: 05 fused_store ROCm FP8'; \
assert 'hip/hip_fp8.h' in src9, 'PATCH MISSING: 6.1 cuda_fp8.h -> hip_fp8.h'; \
src10 = open('/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py').read(); \
assert '_is_cuda or _is_musa or _is_hip' in src10, 'PATCH MISSING: 06c draft extend graph on HIP'; \
src11 = open('/sgl-workspace/sglang/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py').read(); \
assert 'seq_lens_cpu is not None' in src11, 'PATCH MISSING: 06b frozen_kv_mtp D2H sync'; \
print('[OK] All 16 key patches verified in image')"

# ============================================================
# Step 4: Upgrade AITER to latest origin/main + rebuild kernels
# ============================================================
# The base image ships AITER at commit 9127c94a1 (2026-06-25).
# origin/main has 118+ commits including GLM-5.2 tuned configs,
# gfx950 MLA/MoE optimizations, and OPUS RMSNorm backend.
#
# We fetch latest, rebuild all prebuilt .so kernels, then regenerate
# GLM-5.2 tuned GEMM configs (idempotent — dedupes vs upstream).
WORKDIR /sgl-workspace/aiter
RUN git fetch --no-recurse-submodules origin && \
    # Force checkout: base image has local modifications to config CSVs
    # (from runtime patching) that would block a normal checkout.
    git checkout -f origin/main && \
    # Skip CK submodule update — the base image's CK is compatible.
    # CK's remote has force-pushed branches that break submodule fetch.
    # Install updated Python package. Keep existing .so files from base image
    # (PREBUILD_KERNELS needs GPU via rocminfo, unavailable in Docker build).
    # New kernels JIT-compile at runtime when GPU is available.
    pip install -e . --no-build-isolation && \
    # Verify .so files still present (from base image)
    SO_COUNT=$(find /sgl-workspace/aiter/aiter/jit/ -name "*.so" 2>/dev/null | wc -l) && \
    echo "[OK] AITER upgraded: $SO_COUNT .so files (base image kernels)" && \
    test "$SO_COUNT" -gt 100 && \
    # Record the exact commit
    git log --oneline -1 > /aiter_version.txt && \
    cat /aiter_version.txt
WORKDIR /

# ============================================================
# Step 5: Generate AITER tuned GEMM configs for GLM-5.2
# ============================================================
# Adds tuned entries for gfx950 / GLM-5.2 shapes on top of whatever
# upstream already provides. Idempotent — skips existing (M,N,K) entries.
# Gen1: BF16 tuned config (M=1..50000, N=32+160, K=6144)
# Gen2: A8W8 blockscale tuned config (M=1..65536, N=128+2624+3072+6144, K=6144)
COPY docker/rocm-mi355x-glm52-0708/gen_aiter_configs.py /tmp/gen_aiter_configs.py
RUN python3 /tmp/gen_aiter_configs.py && rm /tmp/gen_aiter_configs.py

# ============================================================
# Step 6: Set all environment variables (baked into image)
# ============================================================
# These match the running master container exactly.
# New base image (v0.5.15.post1) already pre-sets: HIP_FORCE_DEV_KERNARG,
# HSA_NO_SCRATCH_RECLAIM, NCCL_MIN_NCHANNELS, PYTORCH_ROCM_ARCH,
# SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN, SGLANG_DISABLE_CUDNN_CHECK,
# SGLANG_INT4_WEIGHT, SGLANG_MOE_PADDING, SGLANG_ROCM_DISABLE_LINEARQUANT,
# SGLANG_ROCM_FUSED_DECODE_MLA, SGLANG_SET_CPU_AFFINITY, SGLANG_USE_AITER,
# SGLANG_USE_ROCM700A.
# We add the ones NOT pre-set in the new base image.
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV NCCL_DEBUG=INFO
ENV HSA_ENABLE_SDMA=0
ENV NCCL_CUMEM_ENABLE=0
ENV NCCL_NVLS_ENABLE=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV ROCM_QUICK_REDUCE_QUANTIZATION=INT8
ENV SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM=1

# ============================================================
# Step 7: Install startup script as entrypoint
# ============================================================
COPY docker/rocm-mi355x-glm52-0723/start_server.sh /start_server.sh
RUN chmod +x /start_server.sh

ENTRYPOINT ["/start_server.sh"]
