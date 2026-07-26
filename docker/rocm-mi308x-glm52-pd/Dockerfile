# GLM-5.2-FP8 1P1D SGLang worker for AMD MI308X (gfx942)
# Based on official rocm/sgl-dev:v0.5.15.post1-rocm700-mi30x-20260715 (ROCm 7.0.0).
#
# This image KEEPS all previous optimizations while upgrading to the newer post1 base.
# Because post1 refactored runtime APIs (removed get_global_server_args, new import
# paths), we CANNOT overlay whole files from the old branch — instead we apply the
# SEMANTIC patch of each change to the post1 source via idempotent patch scripts.
#
# Applied layers:
#   1. Mooncake@01d1eb2a GDR (GPUDirect RDMA via HIP dmabuf) patches + rebuild .so.
#      Uses USE_HIP_DMABUF=ON path (hsa_amd_portable_export_dmabuf + ibv_reg_dmabuf_mr)
#      for true GPU-direct RDMA — NO host staging. Also respects MC_DISABLE_HIP_TRANSPORT=1
#      (disables intra-node HIP IPC transport for cross-node PD).
#      Base already ships mooncake source @01d1eb2a + a configured build dir; no git clone.
#   2. 11 GLM-5.2/PD/HIP patches ported to post1 (scripts under patches/post1/):
#        deepseek_v2.py     (2.1  PCG_DSV2_DUAL_STREAM on HIP)
#        dsa_indexer.py     (1.6  k_norm fp32 cast; 04 breakable-target-verify guard)
#        deepseek_nextn.py  (3.1  alt_stream on HIP)
#        transformers.py    (4.1  mla_kv_a_proj replicate TP style)
#        dsa_backend.py     (5c  MLA q view->reshape on HIP)
#        elementwise.py     (03  q_fp8_raw uint8-then-view on HIP)
#        fused_store_index_cache.cuh (05/6.1 pack_fp8 via deepseek_v4::fp8)
#        eagle_worker_v2.py (06c DSA backend on HIP)
#        frozen_kv_mtp_worker_v2.py (06b seq_lens_cpu fast path)
#        forward_mha.py     (07  aiter+HIP non-power-of-2 nope_dim guard for GLM-5.2)
#        aiter_backend.py   (08  forward_extend/decode +**kwargs for DSA topk_indices)
#   3. FlyDSL gfx942 fp8 MQA logits kernel (BLOCK_KV=128, register-based)
#   4. AITER tuned GEMM/MoE CSV configs for gfx942
#   5. libbnxt_re-rdmav34.so 238.1.138.5 (ts4 kernel bnxt_re ABI match; image default 235.x fails)
#
# Build (from repo root):
#   docker build -f docker/rocm-mi308x-glm52-pd/Dockerfile \
#     -t sglang-glm52-308x-pd:rocm700-0716-gdr .

FROM rocm/sgl-dev:v0.5.15.post1-rocm700-mi30x-20260715

LABEL maintainer="guofutan"
LABEL sglang-base-image="v0.5.15.post1-rocm700-mi30x-20260715"
LABEL gpu-arch="gfx942"
LABEL rdma-mode="GDR (HIP dmabuf, no host staging)"

# ============================================================
# Layer 1: Apply Mooncake GDR patches + rebuild .so
# Base already ships Mooncake@01d1eb2a source at /sgl-workspace/Mooncake with a
# pre-configured build dir (USE_HIP_DMABUF=ON in CMakeCache.txt), and engine.so is
# pip-installed (non-editable) into site-packages. We strip any prior host-staging
# injection, add MC_DISABLE_HIP_TRANSPORT support, then rebuild + copy the .so.
# The resulting engine.so uses the GDR path:
#   hipMemoryTypeDevice + isKernelDmabufSupported()
#     -> hsa_amd_portable_export_dmabuf() + ibv_reg_dmabuf_mr()
# NOT host staging (hipMallocHost + hipMemcpy + ibv_reg_mr).
# ============================================================
COPY docker/rocm-mi308x-glm52-pd/patches/mooncake/apply_gdr.sh /tmp/apply_gdr.sh
RUN set -eux; \
    chmod +x /tmp/apply_gdr.sh && cd /sgl-workspace/Mooncake && /tmp/apply_gdr.sh; \
    cd /sgl-workspace/Mooncake/build && \
    find . -name "*.o" -delete && \
    find . -name "*.a" -delete && \
    cmake . 2>&1 | grep -iE "HIP dmabuf|hsa-runtime64" || true; \
    make -j "$(nproc)" && \
    cp mooncake-integration/engine.cpython-310-x86_64-linux-gnu.so \
       /opt/venv/lib/python3.10/site-packages/mooncake/engine.cpython-310-x86_64-linux-gnu.so && \
    ldconfig; \
    echo "[Mooncake] GDR patches applied and .so refreshed (USE_HIP_DMABUF path)"; \
    # Verify the GDR dmabuf path was actually compiled into the .so. If this
    # fails, the CMake USE_HIP_DMABUF compile definition did not propagate to
    # the rdma_transport target and the GDR path is silently disabled.
    DMABUF_REFS=$(strings /opt/venv/lib/python3.10/site-packages/mooncake/engine.cpython-310-x86_64-linux-gnu.so | grep -c dmabuf); \
    if [ "$DMABUF_REFS" -lt 5 ]; then \
        echo "ERROR: engine.so has only $DMABUF_REFS dmabuf refs (expected >=5). GDR is NOT compiled in."; \
        echo "CMake CXX_DEFINES for rdma_transport:"; \
        cat /sgl-workspace/Mooncake/build/mooncake-transfer-engine/src/transport/rdma_transport/CMakeFiles/rdma_transport.dir/flags.make 2>/dev/null | grep DEFINES; \
        exit 1; \
    fi; \
    echo "[Mooncake] VERIFIED: engine.so has $DMABUF_REFS dmabuf refs (GDR compiled in)"

# ============================================================
# Layer 2: Overlay full sglang Python source from repo.
# This replaces the post1 base's sglang source with our working source
# (which includes all GLM-5.2/PD/HIP patches + Anthropic PD bootstrap fields).
# The old working image (api-fix-0720-gdr-v3) used this same approach.
# ============================================================
COPY python/sglang/ /sgl-workspace/sglang/python/sglang/

# ============================================================
# Layer 2b: Fix import path for template_detection.
# Our source uses sglang.srt.managers.template_detection but the post1
# base image has it at sglang.srt.parser.template_detection. The full
# source overlay above copies our managers/ dir but the parser/ dir from
# the base may have been removed. Add a shim to redirect the import.
# ============================================================
RUN sed -i 's/from sglang\.srt\.managers\.template_detection/from sglang.srt.parser.template_detection/' \
    /sgl-workspace/sglang/python/sglang/srt/entrypoints/anthropic/serving.py 2>/dev/null || true; \
    # If parser/template_detection.py doesn't exist, copy it from managers/
    if [ ! -f /sgl-workspace/sglang/python/sglang/srt/parser/template_detection.py ]; then \
        cp /sgl-workspace/sglang/python/sglang/srt/managers/template_detection.py \
           /sgl-workspace/sglang/python/sglang/srt/parser/template_detection.py 2>/dev/null || true; \
    fi; \
    python3 -c "\
import pathlib as p; f=p.Path('/sgl-workspace/sglang/python/sglang/srt/entrypoints/anthropic/protocol.py').read_text(); \
assert 'bootstrap_host' in f and 'disagg_prefill_dp_rank' in f, 'protocol.py bootstrap fields missing'; \
f2=p.Path('/sgl-workspace/sglang/python/sglang/srt/entrypoints/anthropic/serving.py').read_text(); \
assert 'bootstrap_host' in f2 and 'disagg_prefill_dp_rank' in f2, 'serving.py bootstrap passthrough missing'; \
print('[OK] Anthropic PD bootstrap fields installed')"

# ============================================================
# Layer 2c: Apply patches NOT present in our source tree.
# The full source overlay (Layer 2) covers most patches, but two were
# only in the patch scripts and not in the source tree:
#   07: forward_mha.py — power-of-2 guard for aiter+HIP (GLM-5.2 nope_dim=192)
#   08: aiter_backend.py — **kwargs in forward_extend/forward_decode (DSA topk_indices)
# Both scripts are idempotent (skip if marker already present).
# ============================================================
COPY docker/rocm-mi308x-glm52-pd/patches/post1/forward_mha.py /tmp/patch_forward_mha.py
COPY docker/rocm-mi308x-glm52-pd/patches/post1/aiter_backend.py /tmp/patch_aiter_backend.py
RUN set -eux; \
    python3 /tmp/patch_forward_mha.py \
      /sgl-workspace/sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py; \
    python3 /tmp/patch_aiter_backend.py \
      /sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py; \
    rm /tmp/patch_forward_mha.py /tmp/patch_aiter_backend.py; \
    echo "[OK] Patches 07 (forward_mha np2 guard) + 08 (aiter_backend **kwargs) applied"

# Verify patches are present in the overlaid source.
RUN python3 -c "\
import pathlib as p; R=p.Path('/sgl-workspace/sglang/python/sglang'); r=lambda x:(R/x).read_text(); \
assert '(_is_cuda or _is_hip) and envs.SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM' in r('srt/models/deepseek_v2.py'), '2.1'; \
assert '_k_norm_weight_f32' in r('srt/layers/attention/dsa/dsa_indexer.py'), '1.6'; \
assert 'FIX(breakable-target-verify)' in r('srt/layers/attention/dsa/dsa_indexer.py'), '04'; \
assert 'is_hip() or envs.SGLANG_NPU_USE_MULTI_STREAM' in r('srt/models/deepseek_nextn.py'), '3.1'; \
assert '\"mla_kv_a_proj\": \"replicate\"' in r('srt/models/transformers.py'), '4.1'; \
assert 'q.reshape(-1, layer.tp_q_head_num, layer.v_head_dim)' in r('srt/layers/attention/dsa_backend.py'), '5c'; \
assert 'q_fp8_raw' in r('jit_kernel/dsv4/elementwise.py'), '03'; \
assert 'deepseek_v4::fp8::pack_fp8' in r('jit_kernel/csrc/dsa/fused_store_index_cache.cuh'), '05/6.1'; \
assert '_np2_aiter_guard' in r('srt/models/deepseek_common/attention_forward_methods/forward_mha.py'), '07'; \
assert '_dsa_kwargs_compat' in r('srt/layers/attention/aiter_backend.py'), '08'; \
assert '_is_cuda or _is_musa or _is_hip' in r('srt/speculative/eagle_worker_v2.py'), '06c'; \
assert 'batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()' in r('srt/speculative/frozen_kv_mtp_worker_v2.py'), '06b'; \
assert 'DUAL_STREAM_TOKEN_THRESHOLD = 1024' in r('srt/layers/attention/dsa/dsa_indexer.py'), '06a base'; \
assert 'is_shuffled = True' in r('srt/layers/quantization/fp8.py'), '02 base'; \
print('[OK] All patches verified in overlaid source')"

# ============================================================
# Layer 3: FlyDSL gfx942 fp8 MQA logits kernel (308x 64KB shared mem limit)
# + Triton fp8_mqa_logits FN→FNUZ fix + BLOCK_KV=64 fallback
# + Paged pa_mqa_logits ChunkK=128 (was 256, exceeds 64KB shared mem on gfx942)
# ============================================================
COPY docker/rocm-mi308x-glm52/patches/flydsl/fp8_mqa_logits.py /sgl-workspace/aiter/aiter/ops/flydsl/kernels/fp8_mqa_logits.py
COPY docker/rocm-mi308x-glm52/patches/flydsl/__init__.py /sgl-workspace/aiter/aiter/ops/flydsl/__init__.py
COPY docker/rocm-mi308x-glm52/patches/fp8_mqa_logits.py /sgl-workspace/aiter/aiter/ops/triton/attention/fp8_mqa_logits.py
# Patch paged MQA logits: reduce ChunkK from 256 to 64 for gfx942.
# The _deepgemm_fp8_paged_mqa_logits kernel at pa_mqa_logits.py:348 fails in
# ConvertTritonAMDGPUToLLVM when ChunkK=256 or 128. ChunkK=64 is already proven
# to work (used in the ragged_k variant at line 90).
RUN sed -i 's/ChunkK: int = 256/ChunkK: int = 64/g; s/ChunkK: int = 128/ChunkK: int = 64/g' \
    /sgl-workspace/aiter/aiter/ops/triton/attention/pa_mqa_logits.py && \
    grep -n 'ChunkK: int = ' /sgl-workspace/aiter/aiter/ops/triton/attention/pa_mqa_logits.py | head -5 && \
    echo "[OK] pa_mqa_logits ChunkK patched to 64 for gfx942"

# ============================================================
# Layer 4: AITER tuned GEMM/MoE configs for gfx942 (pre-built csv)
# ============================================================
COPY docker/rocm-mi308x-glm52/patches/glm5_1_gemm_gfx942.csv /sgl-workspace/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv
COPY docker/rocm-mi308x-glm52/patches/glm5_1_fmoe_gfx942.csv /sgl-workspace/aiter/aiter/configs/model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv

# ============================================================
# Layer 5: libbnxt_re-rdmav34.so 238.1.138.5 (ts4 kernel bnxt_re ABI match)
# The base image ships libbnxt_re 235.2.86.0, which does NOT support the ts4
# kernel's bnxt_re ABI (kernel reports ABI 1, driver expects 7-8), so
# ibv_devinfo shows "No IB devices found". The 238.x driver matches the ts4
# kernel module ABI and is required for RDMA on ts4 nodes.
# ============================================================
COPY docker/rocm-mi308x-glm52-pd/patches/libbnxt_re/libbnxt_re-rdmav34.so /tmp/libbnxt_re-rdmav34.so
RUN set -eux; \
    cp /tmp/libbnxt_re-rdmav34.so /usr/local/lib/libbnxt_re-rdmav34.so && \
    cp /tmp/libbnxt_re-rdmav34.so /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav34.so && \
    ldconfig && \
    rm /tmp/libbnxt_re-rdmav34.so && \
    echo "[libbnxt_re] installed 238.1.138.5 (ts4 kernel ABI match)" && \
    strings /usr/local/lib/libbnxt_re-rdmav34.so | grep -m1 "238\."

# ============================================================
# Layer 6: Environment (gfx942 + 308x-specific)
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
