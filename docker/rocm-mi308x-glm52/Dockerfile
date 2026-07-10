# GLM-5.2-FP8 SGLang Worker for AMD MI308X (gfx942)
# Based on official mi30x ROCm image with 0708-opt patches applied.
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
# All code patches from 0708-opt branch are baked in — no runtime patching needed.
# Environment variables are pre-set via ENV.
# The entrypoint (start_server.sh) only needs MODEL_PATH, API_KEY, PORT.

FROM lmsysorg/sglang-rocm:v0.5.14-rocm720-mi30x-20260708

LABEL maintainer="guofutan"
LABEL description="GLM-5.2-FP8 SGLang worker for AMD MI308X (gfx942) with 0708-opt patches"
LABEL sglang-base-image="v0.5.14-rocm720-mi30x-20260708"
LABEL gpu-arch="gfx942"
LABEL branch="308x-glm52-opt"

# ============================================================
# Step 1: Replace sglang source with pre-patched version (0708-opt)
# ============================================================
# The official image has sglang installed in editable mode pointing to
# /sgl-workspace/sglang/python/sglang/. We overwrite the source files
# with our pre-patched version from the 0708-opt branch.
# This fixes cooperative_groups.h, cuda_fp8.h, and other HIP compatibility issues.
COPY python/sglang/ /sgl-workspace/sglang/python/sglang/

# Verify key patches are present (fail build if any missing)
# Also verify 308X-specific workarounds are REMOVED (matching 355X config)
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
fs = open('/sgl-workspace/sglang/python/sglang/jit_kernel/fused_store_index_cache.py').read(); \
assert 'DISABLED on ROCm' not in fs, 'REVERT FAILED: fused-store is_hip guard still present'; \
assert 'if is_hip():' not in fs, 'REVERT FAILED: fused-store is_hip() check still present'; \
db = open('/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py').read(); \
assert 'and not _is_hip' not in db.split('_USE_FUSED_METADATA_COPY')[1].split(')')[0] if '_USE_FUSED_METADATA_COPY' in db else True, 'REVERT FAILED: fused metadata copy still disabled on HIP'; \
print('All 16 patches verified + 308X workarounds reverted OK')"

# ============================================================
# Step 1b: Patch AITER fp8_mqa_logits for gfx942 shared memory limit
# MI308X (gfx942) has 64KB shared memory limit vs MI355X's 80KB+.
# Original BLOCK_KV=128 → 80KB shared memory (exceeds 64KB → OutOfResources).
# Patched: BLOCK_KV=64, num_stages=1 → fits in 64KB.
# This is a HARDWARE LIMIT, not a precision tradeoff — the kernel math is identical,
# just smaller tile size. No precision loss.
# ============================================================
COPY docker/rocm-mi308x-glm52/patches/fp8_mqa_logits.py /sgl-workspace/aiter/aiter/ops/triton/attention/fp8_mqa_logits.py

# ============================================================
# Step 2: Environment Variables (21, based on MI355X production config)
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
