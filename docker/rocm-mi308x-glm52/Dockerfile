# GLM-5.2-FP8 SGLang Worker for AMD MI308X (gfx942)
# Based on official mi30x ROCm image with upstream main + EAGLE coredump fixes.
#
# Build (from repo root):
#   docker build -f docker/rocm-mi308x-glm52/Dockerfile \
#     -t sglang-glm52-308x:fix-eagle-coredump .
#
# Run:
#   docker run -d --name sglang_308x \
#     --privileged --network host --shm-size 32g \
#     --device /dev/kfd --device /dev/dri --group-add video \
#     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
#     -v /data:/data \
#     sglang-glm52-308x:fix-eagle-coredump
#
# This branch (fix/eagle-decode-coredump-mi308x) is based on upstream main
# (b3570a4531, 2026-07-20) which contains 380 commits beyond the previous
# container base (b76dd0be, 2026-07-10), including critical EAGLE fixes:
#   - 78dc581518 Stabilize GLM-5.2 MTP IndexShare across PD and CUDA graph replay
#   - 7a973c03a0 Stamp capture-time num_tokens_per_req in multi-layer EAGLE
#   - fc1e3797b7 Split capture width from num_tokens_per_req and gate replay
#   - cce5fe7696 Move WAR barrier right after each run_batch launch
#   - 942bf04ef9 Add SGLANG_FORCE_COARSE_WAR_BARRIER opt-in
# Plus our local patch:
#   - PR #31478: TP broadcast in EAGLE greedy branch (not yet merged upstream)
#
# The entrypoint (start_server.sh) only needs MODEL_PATH, API_KEY, PORT.

FROM lmsysorg/sglang-rocm:v0.5.15.post1-rocm720-mi30x-20260718

LABEL maintainer="guofutan"
LABEL description="GLM-5.2-FP8 SGLang worker for AMD MI308X (gfx942) with EAGLE coredump fix"
LABEL sglang-base-image="v0.5.15.post1-rocm720-mi30x-20260718"
LABEL gpu-arch="gfx942"
LABEL branch="fix/eagle-decode-coredump-mi308x"
LABEL upstream-base="b3570a4531 (origin/main, 2026-07-20)"
LABEL local-patch="PR #31478 TP broadcast in EAGLE greedy branch"

# ============================================================
# Step 1: Replace sglang source with upstream main + local patches
# ============================================================
# The official image has sglang installed in editable mode pointing to
# /sgl-workspace/sglang/python/sglang/. We overwrite the source files
# with our fix branch version (upstream main + PR #31478).
COPY python/sglang/ /sgl-workspace/sglang/python/sglang/

# Verify the EAGLE coredump fix is present (fail build if missing)
RUN python3 -c "\
eagle = open('/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_utils.py').read(); \
assert 'PR #31478' in eagle, 'PATCH MISSING: PR #31478 TP broadcast in greedy branch'; \
assert 'tp_group.broadcast(predict, src=0)' in eagle, 'PATCH MISSING: TP broadcast call'; \
ml_runner = open('/sgl-workspace/sglang/python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py').read(); \
assert 'captured_req_width' in ml_runner, 'UPSTREAM FIX MISSING: fc1e3797b7 captured_req_width'; \
assert 'spec_info.num_tokens_per_req = self.captured_req_width' in ml_runner, 'UPSTREAM FIX MISSING: 7a973c03a0 stamp num_tokens_per_req at capture'; \
eagle_info = open('/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_info.py').read(); \
assert 'future_dsa_topk_indices_available' in eagle_info, 'UPSTREAM FIX MISSING: 78dc581518 dsa_topk_indices stabilization'; \
eagle_worker = open('/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py').read(); \
assert 'seed_dsa_topk_from_draft_extend' in eagle_worker, 'UPSTREAM FIX MISSING: 78dc581518 seed_dsa_topk guard'; \
sched = open('/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py').read(); \
assert 'SGLANG_FORCE_COARSE_WAR_BARRIER' in sched, 'UPSTREAM FIX MISSING: 942bf04ef9 coarse WAR barrier'; \
assert '_apply_war_barrier' in sched, 'UPSTREAM FIX MISSING: cce5fe7696 WAR barrier after run_batch'; \
print('All EAGLE coredump fixes verified: PR #31478 + 78dc581518 + 7a973c03a0 + fc1e3797b7 + cce5fe7696 + 942bf04ef9')"

# ============================================================
# Step 1b: Add FlyDSL gfx942 fp8 MQA logits kernel + patch Triton path
# MI308X (gfx942) has 64KB shared memory limit vs MI355X's 80KB+.
# Strategy: Use FlyDSL kernel (designed for gfx942, adaptive grid split)
# + patch Triton fp8_mqa_logits with FN->FNUZ conversion fix + BLOCK_KV=64
# The FlyDSL kernel uses _BLOCK_KV=128 but with different memory layout
# (waves_per_eu=2, no cross-wave LDS) that may fit in 64KB.
# The Triton path is patched as fallback with BLOCK_KV=64.
# ============================================================
# Add FlyDSL kernel files
COPY docker/rocm-mi308x-glm52/patches/flydsl/fp8_mqa_logits.py /sgl-workspace/aiter/aiter/ops/flydsl/kernels/fp8_mqa_logits.py
COPY docker/rocm-mi308x-glm52/patches/flydsl/__init__.py /sgl-workspace/aiter/aiter/ops/flydsl/__init__.py
# Patch Triton fp8_mqa_logits with FN->FNUZ fix + BLOCK_KV=64 (fallback)
COPY docker/rocm-mi308x-glm52/patches/fp8_mqa_logits.py /sgl-workspace/aiter/aiter/ops/triton/attention/fp8_mqa_logits.py

# ============================================================
# Step 2: Environment Variables (based on MI355X production config)
# Key change: PYTORCH_ROCM_ARCH=gfx942 (was gfx950 for MI355X)
#
# Updates for EAGLE coredump fix (2026-07-20):
#   - NCCL_DEBUG=WARN (was INFO): suppress INFO log spam that slows health probe
#   - NCCL_MIN_NCHANNELS=80 (was 112): RCCL 2.27.7 hard cap is 80, higher values
#     get silently clamped and spam INFO logs
#   - SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false: /health returns 200 directly
#     instead of running prefill 64 tokens, avoids GPU wake-up delay
#   - SGLANG_FORCE_COARSE_WAR_BARRIER=true: force whole-forward WAR barrier,
#     more stable for EAGLE overlap scheduling on ROCm
#   - ROCM_QUICK_REDUCE_QUANTIZATION=NONE (was INT8): zero precision loss,
#     activation memory 2.5GB -> 46GB, gen_throughput +24%
# ============================================================
ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV NCCL_DEBUG=WARN
ENV HSA_ENABLE_SDMA=0
ENV HIP_FORCE_DEV_KERNARG=1
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV NCCL_CUMEM_ENABLE=0
ENV NCCL_MIN_NCHANNELS=80
ENV NCCL_NVLS_ENABLE=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYTORCH_ROCM_ARCH=gfx942
ENV ROCM_QUICK_REDUCE_QUANTIZATION=NONE
ENV SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
ENV SGLANG_DISABLE_CUDNN_CHECK=1
ENV SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM=1
ENV SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false
ENV SGLANG_FORCE_COARSE_WAR_BARRIER=true
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
