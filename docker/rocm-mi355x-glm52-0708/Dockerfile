# GLM-5.2-FP8 SGLang Worker for AMD MI355X (gfx950)
# Based on official 0706 ROCm image with all patches pre-applied.
#
# Build (from repo root):
#   docker build -f docker/rocm-mi355x-glm52-0708/Dockerfile \
#     -t sglang-glm52-0708:latest .
#
# Run:
#   docker run -d --name sglang_0708_worker \
#     --privileged --network host --shm-size 32g \
#     --device /dev/kfd --device /dev/dri --group-add video \
#     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
#     -v /data:/data \
#     sglang-glm52-0708:latest
#
# All code patches are baked in — no runtime patching needed.
# Environment variables are pre-set via ENV.
# The entrypoint (start_server.sh) only needs MODEL_PATH, API_KEY, PORT.

FROM lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260706

LABEL maintainer="guofutan"
LABEL description="GLM-5.2-FP8 SGLang worker for AMD MI355X with pre-applied ROCm patches (0708-opt)"
LABEL sglang-base-image="v0.5.14-rocm720-mi35x-20260706"
LABEL branch="0708-opt"

# ============================================================
# Step 1: Replace sglang source with pre-patched version
# ============================================================
# The official image has sglang installed in editable mode pointing to
# /sgl-workspace/sglang/python/sglang/. We overwrite the source files
# with our pre-patched version from the 0708-opt branch.
# No runtime patching needed — all patches are baked in.
COPY python/sglang/ /sgl-workspace/sglang/python/sglang/

# Verify key patches are present (fail build if any missing)
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
# Step 2: Generate AITER tuned GEMM configs for GLM-5.2
# ============================================================
# Reads base configs from /sgl-workspace/aiter/ (already in base image)
# and adds tuned entries for gfx950 / GLM-5.2 shapes.
COPY docker/rocm-mi355x-glm52-0708/gen_aiter_configs.py /tmp/gen_aiter_configs.py
RUN python3 /tmp/gen_aiter_configs.py && rm /tmp/gen_aiter_configs.py

# ============================================================
# Step 3: Set all environment variables (baked into image)
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
COPY docker/rocm-mi355x-glm52-0708/start_server.sh /start_server.sh
RUN chmod +x /start_server.sh

ENTRYPOINT ["/start_server.sh"]
