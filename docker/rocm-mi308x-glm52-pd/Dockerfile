# GLM-5.2-FP8 1P1D SGLang worker for AMD MI308X (gfx942)
# Based on official latest mi30x v0.5.15 image + Mooncake@01d1eb2a + 16 GLM-5.2 patches + FlyDSL gfx942.
# GPU-direct RDMA (GDR) enabled: USE_HIP_DMABUF propagated to rdma_transport target.
#
# Build (from repo root):
#   docker build -f docker/rocm-mi308x-glm52-pd/Dockerfile -t sglang-glm52-308x-pd:gdr-0720 .
#
# PD deploy via kubectl (see yamls/). GDR eliminates the 43.2 GiB per-request
# hipMemcpy bottleneck — KV cache transfers go directly GPU-to-GPU via RDMA.

FROM lmsysorg/sglang-rocm:v0.5.15-rocm720-mi30x-20260710

LABEL maintainer="guofutan"
LABEL sglang-base-image="v0.5.15-rocm720-mi30x-20260710"
LABEL gpu-arch="gfx942"

# ============================================================
# Hack 1: Rebuild Mooncake@01d1eb2a (main HEAD has PD warmup SIGSEGV regression)
# ============================================================
ARG MOONCAKE_REF=01d1eb2a7ec37fd5e20a88573e9b4956e7846e9a
ARG MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake.git"
ENV PATH=${PATH}:/usr/local/go/bin
RUN set -eux; \
    rm -rf /sgl-workspace/Mooncake; \
    git clone --depth 1 "${MOONCAKE_REPO}" /sgl-workspace/Mooncake; \
    cd /sgl-workspace/Mooncake; \
    git fetch --depth 1 origin "${MOONCAKE_REF}"; \
    git checkout "${MOONCAKE_REF}"; \
    echo "[Mooncake] ref=${MOONCAKE_REF} commit=$(git rev-parse HEAD)"; \
    git submodule update --init --recursive; \
    bash dependencies.sh -y; \
    mkdir -p build && cd build; \
    cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON -DENABLE_MULTI_PROTOCOL=ON -DWITH_STORE=ON -DBUILD_UNIT_TESTS=OFF; \
    make -j "$(nproc)" && make install; ldconfig; \
    echo "[Mooncake] rebuild complete @ ${MOONCAKE_REF}"

# ============================================================
# Hack 1b: Apply Mooncake C++ patches for bnxt_re (MC_DISABLE_HIP_TRANSPORT + staging fields)
# ============================================================
COPY docker/rocm-mi308x-glm52-pd/patches/mooncake/apply_host_staging.sh /tmp/apply_host_staging.sh
RUN set -eux; \
    chmod +x /tmp/apply_host_staging.sh && /tmp/apply_host_staging.sh; \
    echo "[Mooncake] staging patches applied"

# ============================================================
# Hack 1c: Enable GPU-direct RDMA (GDR) — fix USE_HIP_DMABUF propagation
# Root cause: USE_HIP_DMABUF is PRIVATE on transfer_engine, doesn't reach
# rdma_transport OBJECT library where rdma_context.cpp lives. Without this fix,
# rdma_context.cpp compiles the #else host-staging branch (43.2 GiB hipMemcpy
# per request). With this fix, it compiles the USE_HIP_DMABUF branch which
# falls back to ibv_reg_mr(pd, gpu_ptr, ...) when kernel lacks P2PDMA —
# verified to work end-to-end on MI308X + bnxt_re (RDMA write data integrity confirmed).
# ============================================================
COPY docker/rocm-mi308x-glm52-pd/patches/mooncake/enable_gdr.sh /tmp/enable_gdr.sh
RUN set -eux; \
    chmod +x /tmp/enable_gdr.sh && /tmp/enable_gdr.sh; \
    cd /sgl-workspace/Mooncake/build && \
    find . -name "*.o" -delete && \
    cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON -DENABLE_MULTI_PROTOCOL=ON -DWITH_STORE=ON -DBUILD_UNIT_TESTS=OFF && \
    make -j "$(nproc)" && make install; ldconfig; \
    echo "[Mooncake] GPU-direct RDMA (GDR) enabled — USE_HIP_DMABUF propagated to rdma_transport"

# Verify GDR fix: engine.so must contain dmabuf symbols (NOT host staging)
RUN strings /opt/venv/lib/python3.10/site-packages/mooncake/engine.cpython-310-x86_64-linux-gnu.so | \
    grep -q "isKernelDmabuf\|hsa_amd_portable_export_dmabuf" && \
    echo "[GDR VERIFY] dmabuf symbols present in engine.so" || \
    (echo "[GDR VERIFY FAILED] dmabuf symbols NOT found — GDR fix did not work"; exit 1); \
    strings /opt/venv/lib/python3.10/site-packages/mooncake/engine.cpython-310-x86_64-linux-gnu.so | \
    grep -q "RDMA host staging: GPU=" && \
    echo "[GDR WARNING] host staging string still present (may be from #else dead code)" || \
    echo "[GDR VERIFY] host staging string absent — clean GDR build"

# ============================================================
# Hack 2: Overlay patched sglang source (GLM-5.2 + PD + HIP fixes, 0708-opt branch)
# ============================================================
COPY python/sglang/ /sgl-workspace/sglang/python/sglang/

# Verify 16 key patches + 308x revert checks
RUN python3 -c "\
src = open('/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py').read(); \
assert '_is_cuda or _is_hip' in src, 'PATCH MISSING: 1.1'; \
assert '(_is_cuda or _is_hip) and not envs.SGLANG_DISABLE_DSA_INDEXER_FUSION' in src, 'PATCH MISSING: 1.2'; \
assert '_k_norm_weight_f32' in src, 'PATCH MISSING: 1.6'; \
assert 'FIX(breakable-target-verify)' in src, 'PATCH MISSING: 04'; \
assert 'DUAL_STREAM_TOKEN_THRESHOLD = 1024' in src, 'PATCH MISSING: 06a'; \
src2 = open('/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py').read(); \
assert '(_is_cuda or _is_hip) and envs.SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM' in src2, 'PATCH MISSING: 2.1'; \
src3 = open('/sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py').read(); \
assert 'is_hip() or envs.SGLANG_NPU_USE_MULTI_STREAM' in src3, 'PATCH MISSING: 3.1'; \
src4 = open('/sgl-workspace/sglang/python/sglang/srt/models/transformers.py').read(); \
assert 'mla_kv_a_proj' in src4, 'PATCH MISSING: 4.1'; \
src5 = open('/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py').read(); \
assert 'q.reshape(-1, layer.tp_q_head_num' in src5, 'PATCH MISSING: 5c'; \
assert 'elif forward_batch.seq_lens_sum is not None' in src5, 'PATCH MISSING: 06b'; \
src6 = open('/sgl-workspace/sglang/python/sglang/srt/layers/radix_attention.py').read(); \
assert '# if _is_hip and not save_kv_cache' in src6, 'PATCH MISSING: 5d'; \
src7 = open('/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8.py').read(); \
assert 'is_shuffled = True' in src7, 'PATCH MISSING: 02'; \
src8 = open('/sgl-workspace/sglang/python/sglang/jit_kernel/dsv4/elementwise.py').read(); \
assert 'q_fp8_raw' in src8, 'PATCH MISSING: 03'; \
src9 = open('/sgl-workspace/sglang/python/sglang/jit_kernel/csrc/dsa/fused_store_index_cache.cuh').read(); \
assert 'USE_ROCM' in src9 and 'hip/hip_fp8.h' in src9, 'PATCH MISSING: 05/6.1'; \
src10 = open('/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py').read(); \
assert '_is_cuda or _is_musa or _is_hip' in src10, 'PATCH MISSING: 06c'; \
src11 = open('/sgl-workspace/sglang/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py').read(); \
assert 'seq_lens_cpu is not None' in src11, 'PATCH MISSING: 06b frozen_kv'; \
print('[OK] All 16 GLM-5.2/PD patches verified')"

# ============================================================
# Hack 2b: /v1/responses PD support is now in the source directly.
# protocol.py: ResponsesRequest has bootstrap_host/port/room + disagg_prefill_dp_rank.
# serving_responses.py: bootstrap passthrough + stream=bool + loop-skip in prefill
#   mode + created_at int fix for codex streaming compatibility.
# No patch scripts needed — source is modified in-repo on this branch.
# ============================================================

# ============================================================
# Layer 2b: Apply /v1/messages (Anthropic) PD disaggregation patches.
# Adds bootstrap_* fields to AnthropicMessagesRequest + forwards them to
# chat_request in _convert_to_chat_completion_request so router-injected
# bootstrap_room reaches the prefill worker via GenerateReqInput.
# (Note: /v1/responses PD patches are already in the source tree —
#  commits a48595ba87 + ae127462e8 moved them into the source.)
# ============================================================
COPY docker/rocm-mi308x-glm52-pd/patches/patch_messages_pd.py /tmp/patch_messages_pd.py
RUN set -eux; \
    python3 /tmp/patch_messages_pd.py; \
    python3 -c "\
import pathlib as p; R=p.Path('/sgl-workspace/sglang/python/sglang/srt/entrypoints'); \
assert 'disagg_prefill_dp_rank' in (R/'anthropic/protocol.py').read_text(), 'anthropic protocol'; \
assert 'chat_request.disagg_prefill_dp_rank' in (R/'anthropic/serving.py').read_text(), 'anthropic serving'; \
print('[OK] /v1/messages PD patches verified')"

# ============================================================
# Hack 3: FlyDSL gfx942 fp8 MQA logits kernel (308x 64KB shared mem limit)
# + Triton fp8_mqa_logits FN→FNUZ fix + BLOCK_KV=64 fallback
# ============================================================
COPY docker/rocm-mi308x-glm52/patches/flydsl/fp8_mqa_logits.py /sgl-workspace/aiter/aiter/ops/flydsl/kernels/fp8_mqa_logits.py
COPY docker/rocm-mi308x-glm52/patches/flydsl/__init__.py /sgl-workspace/aiter/aiter/ops/flydsl/__init__.py
COPY docker/rocm-mi308x-glm52/patches/fp8_mqa_logits.py /sgl-workspace/aiter/aiter/ops/triton/attention/fp8_mqa_logits.py

# ============================================================
# Hack 4: AITER tuned GEMM/MoE configs for gfx942 (pre-built csv)
# ============================================================
COPY docker/rocm-mi308x-glm52/patches/glm5_1_gemm_gfx942.csv /sgl-workspace/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv
COPY docker/rocm-mi308x-glm52/patches/glm5_1_fmoe_gfx942.csv /sgl-workspace/aiter/aiter/configs/model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv

# ============================================================
# Hack 5: Environment (gfx942 + 308x-specific)
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
ENV SGLANG_OPT_USE_TOPK_V2=0

CMD ["/bin/bash"]
