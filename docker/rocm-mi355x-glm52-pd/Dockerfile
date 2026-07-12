# GLM-5.2-FP8 SGLang 1P1D (Prefill/Decode disaggregation) worker for AMD MI355X (gfx950)
#
# Strategy:
#   1. FROM the LATEST official sglang ROCm MI355X image
#   2. Rebuild Mooncake from the latest source (overwrites the base image's pinned build)
#   3. Overlay the patched sglang source (origin/main + GLM-5.2/PD/HIP fixes)
#   4. Generate AITER tuned GEMM configs for GLM-5.2 on gfx950
#
# Build (from repo root):
#   docker build -f docker/rocm-mi355x-glm52-pd/Dockerfile \
#     -t sglang-glm52-mi355x-pd:latest .
#
#   Pin Mooncake to its latest release tag instead of main HEAD:
#   docker build --build-arg MOONCAKE_REF=v0.3.11.post1 \
#     -f docker/rocm-mi355x-glm52-pd/Dockerfile -t sglang-glm52-mi355x-pd:latest .
#
# No fixed ENTRYPOINT — prefill and decode take different launch args.
# Pass the full `python3 -m sglang.launch_server ...` command via `docker run`.

FROM lmsysorg/sglang-rocm:v0.5.15-rocm720-mi35x-20260711

LABEL maintainer="guofutan"
LABEL description="GLM-5.2-FP8 1P1D SGLang worker for AMD MI355X — latest official image + latest Mooncake + GLM-5.2 PD patches"
LABEL sglang-base-image="v0.5.15-rocm720-mi35x-20260711"

# ============================================================
# Step 1: Rebuild Mooncake from source
# ============================================================
# The official image ships Mooncake built from commit 01d1eb2a (the version
# sglang v0.5.15 was built/tested against). We rebuild from source so the
# Mooncake version is explicit and patchable.
#
# NOTE: main HEAD (2026-07-12) has a regression — segfaults during PD
# disaggregation warmup (transfer_worker SIGSEGV in mooncake/engine.so).
# Default MOONCAKE_REF=01d1eb2a is the known-working commit. Override to
# `main` or a release tag only after verifying the regression is fixed.
ARG MOONCAKE_REF=01d1eb2a7ec37fd5e20a88573e9b4956e7846e9a
ARG MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake.git"
ENV PATH=${PATH}:/usr/local/go/bin

RUN set -eux; \
    rm -rf /sgl-workspace/Mooncake; \
    if [ "$MOONCAKE_REF" = "main" ]; then \
        git clone --depth 1 "${MOONCAKE_REPO}" /sgl-workspace/Mooncake; \
    else \
        git clone --depth 1 -b "${MOONCAKE_REF}" "${MOONCAKE_REPO}" /sgl-workspace/Mooncake; \
    fi; \
    cd /sgl-workspace/Mooncake; \
    echo "[Mooncake] ref=${MOONCAKE_REF} commit=$(git rev-parse HEAD) date=$(git log -1 --format=%ci)"; \
    git submodule update --init --recursive; \
    bash dependencies.sh -y; \
    mkdir -p build && cd build; \
    cmake .. \
        -DUSE_HIP=ON \
        -DUSE_ETCD=ON \
        -DENABLE_MULTI_PROTOCOL=ON \
        -DWITH_STORE=ON \
        -DBUILD_UNIT_TESTS=OFF; \
    make -j "$(nproc)" && make install; \
    ldconfig; \
    echo "[Mooncake] rebuild complete"

# ============================================================
# Step 2: Overlay patched sglang source (GLM-5.2 + PD + HIP fixes)
# ============================================================
# The official image installs sglang in editable mode at
# /sgl-workspace/sglang/python/sglang/. Overwriting the source files
# updates the importable code without reinstalling (editable install).
# This branch is origin/main + GLM-5.2 DSA/FP8/dual-stream PD fixes
# on HIP — it strictly extends main, so no upstream code is lost.
COPY python/sglang/ /sgl-workspace/sglang/python/sglang/

# Verify the key GLM-5.2 / PD patches are present — fail the build if any
# are missing so a stale source tree never produces a broken image.
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
print('[OK] All 16 key GLM-5.2/PD patches verified in image')"

# ============================================================
# Step 3: Generate AITER tuned GEMM configs for GLM-5.2 on gfx950
# ============================================================
# Reuses the config generator from the 0708 image (same GLM-5.2 shapes).
COPY docker/rocm-mi355x-glm52-0708/gen_aiter_configs.py /tmp/gen_aiter_configs.py
RUN python3 /tmp/gen_aiter_configs.py && rm /tmp/gen_aiter_configs.py

# ============================================================
# Step 4: Environment variables
# ============================================================
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV NCCL_DEBUG=WARN
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

# Prefill/decode launch args differ, so no fixed ENTRYPOINT.
# Run with: docker run ... <image> python3 -m sglang.launch_server ...
CMD ["/bin/bash"]
