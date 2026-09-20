# GLM-5.2/5.3-FP8 1P1D SGLang worker for AMD MI308X (gfx942)
# Based on official rocm/sgl-dev:v0.5.17-rocm720-mi30x-20260809 (ROCm 7.2.0).
#
# SOURCE OF TRUTH: branch mi308x-1p1d-src. The sglang source is committed
# there (base 11d03eaeef + all production patches as real commits, extracted
# from the running v0826 containers on 2026-09-08, parity-diff verified
# against both prefill and decode). This Dockerfile COPYs that source;
# no build-time patching.
#
# Applied layers:
#   1. Mooncake@01d1eb2a GDR (GPUDirect RDMA via HIP dmabuf) patches + rebuild .so.
#      Uses USE_HIP_DMABUF=ON path (hsa_amd_portable_export_dmabuf + ibv_reg_dmabuf_mr)
#      for true GPU-direct RDMA — NO host staging. Also respects MC_DISABLE_HIP_TRANSPORT=1
#      (disables intra-node HIP IPC transport for cross-node PD).
#      Base already ships mooncake source @01d1eb2a + a configured build dir; no git clone.
#   2. sglang source from mi308x-1p1d-src (COPY python/sglang). Includes:
#      14 post1 semantic patches, 9 live patches, dp8ep8 (indexer), fused-topk v2
#      (decode, env-gated), hicache-jit (prefill), 4 prefill runtime patches
#      (evict_backup/prefetch_log/big_node_backup/zombie_free_fix), GDR L2 flush.
#   3. FlyDSL gfx942 fp8 MQA logits kernel (BLOCK_KV=128, register-based)
#   4. AITER tuned GEMM/MoE CSV configs for gfx942
#   5. libbnxt_re-rdmav34.so 238.1.138.5 (ts4 kernel bnxt_re ABI match; image default 235.x fails)
#
# Build (from repo root, on branch mi308x-1p1d-src):
#   docker build -f docker/rocm-mi308x-glm52-pd/Dockerfile \
#     -t mirrors.tencent.com/ti-platform/sglang-glm52-308x:<tag> .

FROM rocm/sgl-dev:v0.5.17-rocm720-mi30x-20260809

LABEL maintainer="guofutan"
LABEL sglang-base-image="v0.5.17-rocm720-mi30x-20260809"
LABEL gpu-arch="gfx942"
LABEL rdma-mode="GDR peermem (SGLANG_PD_HOST_STAGING=0) + optional host-staging fallback"

# ============================================================
# Layer 1: Mooncake true GDR (amdgpu peermem + HipDeviceGuard).
# Do NOT inject C++ host-staging bounce buffers. Python host-staging remains
# env-gated (SGLANG_PD_HOST_STAGING=1) as a fallback.
# ============================================================
COPY docker/rocm-mi308x-glm52-pd/patches/mooncake/apply_gdr.sh /tmp/mooncake/apply_gdr.sh
COPY docker/rocm-mi308x-glm52-pd/patches/mooncake/apply_gdr_peermem.py /tmp/mooncake/apply_gdr_peermem.py
RUN set -eux; \
    chmod +x /tmp/mooncake/apply_gdr.sh && \
    /tmp/mooncake/apply_gdr.sh /sgl-workspace/Mooncake; \
    grep -q 'FIX(gdr-peermem-device-guard)' \
      /sgl-workspace/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp; \
    grep -q 'MC_DISABLE_HIP_TRANSPORT' \
      /sgl-workspace/Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp; \
    cd /sgl-workspace/Mooncake/build && \
    find . -name "*.o" -delete && \
    make -j "$(nproc)" && \
    cp mooncake-integration/engine.cpython-310-x86_64-linux-gnu.so \
       /opt/venv/lib/python3.10/site-packages/mooncake/engine.cpython-310-x86_64-linux-gnu.so && \
    ldconfig; \
    echo "[Mooncake] GDR peermem engine.so rebuilt"

# ============================================================
# Layer 2: sglang source of truth = branch mi308x-1p1d-src.
# The editable install at /sgl-workspace/sglang/python picks this up
# directly. _version.py is build-generated (gitignored) and absent from
# the tree — the base image's generated version survives the COPY.
# ============================================================
COPY python/sglang /sgl-workspace/sglang/python/sglang

# Build-time gate: every production patch marker must be present in the source.
RUN python3 -c "\
import pathlib as p; R=p.Path('/sgl-workspace/sglang/python/sglang'); r=lambda x:(R/x).read_text(); \
assert '(_is_cuda or _is_hip) and envs.SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM' in r('srt/models/deepseek_v2.py'), '2.1'; \
assert '_k_norm_weight_f32' in r('srt/layers/attention/dsa/dsa_indexer.py'), '1.6'; \
assert 'FIX(breakable-target-verify)' in r('srt/layers/attention/dsa/dsa_indexer.py'), '04'; \
assert 'FIX(fused-store-length-guard)' in r('srt/layers/attention/dsa/dsa_indexer.py'), 'fused-store-guard'; \
assert 'i32_safe_rows' in r('srt/layers/attention/dsa/dsa_indexer.py'), 'i32-overflow'; \
assert 'is_hip() or envs.SGLANG_NPU_USE_MULTI_STREAM' in r('srt/models/deepseek_nextn.py'), '3.1'; \
assert '\"mla_kv_a_proj\": \"replicate\"' in r('srt/models/transformers.py'), '4.1'; \
assert 'q.reshape(-1, layer.tp_q_head_num, layer.v_head_dim)' in r('srt/layers/attention/dsa_backend.py'), '5c'; \
assert 'q_fp8_raw' in r('kernels/ops/attention/dsv4/elementwise.py'), '03'; \
assert 'using deepseek_v4::fp8::pack_fp8' in r('kernels/jit/csrc/dsa/fused_store_index_cache.cuh'), '05/6.1'; \
assert '_is_cuda or _is_musa or _is_hip' in r('srt/speculative/eagle_worker_v2.py'), '06c'; \
assert 'batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()' in r('srt/speculative/frozen_kv_mtp_worker_v2.py'), '06b'; \
assert '_dsa_kwargs_compat' in r('srt/layers/attention/aiter_backend.py'), '08'; \
assert 'FIX(bootstrap-list)' in r('srt/disaggregation/decode.py'), 'bootstrap-list'; \
assert 'bootstrap_host' in r('srt/entrypoints/openai/protocol.py'), 'protocol-responses'; \
assert '_fix_created_at_int' in r('srt/entrypoints/openai/serving_responses.py'), 'serving-responses-pd'; \
assert 'FIX(prefill-d2h-host-staging)' in r('srt/disaggregation/mooncake/conn.py'), 'prefill-d2h'; \
assert 'refusing full-pool' in r('srt/disaggregation/mooncake/conn.py'), 'selective-h2d'; \
assert 'FIX(gdr-l2-flush)' in r('srt/disaggregation/mooncake/conn.py'), 'gdr-l2-flush'; \
assert 'rdma_read_flush' in r('srt/disaggregation/mooncake/conn.py'), 'gdr-rdma-read'; \
assert (R/'srt/disaggregation/mooncake/gdr_l2_flush.hsaco').stat().st_size > 0, 'gdr-hsaco'; \
assert 'DUAL_STREAM_TOKEN_THRESHOLD = 1024' in r('srt/layers/attention/dsa/dsa_indexer.py'), '06a base'; \
assert 'is_shuffled = True' in r('srt/layers/quantization/fp8.py'), '02 base'; \
idx = r('srt/layers/attention/dsa/dsa_indexer.py'); \
assert idx.count('flydsl_fp8_mqa_logits') >= 2, 'wave1-flydsl-sites'; \
assert idx.count('WAVE1: FlyDSL gfx942 MQA logits') == 2, 'wave1-flydsl-marker'; \
assert 'self.publish_ready.synchronize()' not in r('srt/managers/overlap_utils.py'), 'wave1-hip-wait'; \
assert 'Event.wait() regresses TPOT' not in r('srt/managers/overlap_utils.py'), 'wave1-no-mi355-sync'; \
assert 'q_fp8[:q_offset]' in idx, 'dp8ep8-hip-q-slice'; \
assert 'weights[:q_offset]' in idx, 'dp8ep8-hip-weights-slice'; \
assert '_use_splitk_topk' in r('srt/layers/attention/dsa/dsa_topk_backend.py'), 'fused-topk-splitk'; \
assert 'should_use_topk_v2' in r('srt/layers/attention/dsa/dsa_topk_backend.py'), 'fused-topk-v2'; \
assert (R/'kernels/ops/attention/dsa/topk_splitk.py').stat().st_size > 0, 'topk-splitk-file'; \
assert 'FIX(evict-backup-v2)' in r('srt/mem_cache/unified_radix_cache.py'), 'evict-backup'; \
assert 'FIX(prefetch-log)' in r('srt/managers/cache_controller.py'), 'prefetch-log'; \
assert 'FIX(big-node-backup)' in r('srt/mem_cache/unified_cache/unified_tree_core.py'), 'big-node-backup'; \
assert 'zombie' in r('srt/mem_cache/unified_cache/unified_tree_core.py').lower(), 'zombie-free'; \
assert (R/'kernels/jit/csrc/kvcacheio/hicache.cuh').stat().st_size > 0, 'hicache-jit'; \
dsa = r('srt/layers/attention/dsa_backend.py'); \
assert 'fused_dsa_decode_metadata' in dsa, 'w1-fused-meta'; \
assert 'is_cuda() and not _is_hip' not in dsa, 'w1-hip-gate'; \
assert 'format_checker=None' in r('srt/function_call/utils.py'), 'tool-schema-regex'; \
assert '_is_glm_moe_dsa_config' in r('srt/models/deepseek_v2.py'), 'w1-bias-fp32'; \
assert '_aiter_fp32_correction_bias_args' in r('srt/layers/moe/topk.py'), 'w1-bias-aiter-upcast'; \
assert 'aiter fp32 correction bias kept' in r('srt/layers/moe/topk.py'), 'w1-bias-once-log'; \
assert 'SGLANG_ROCM_DSA_INDEXER_PROJECTION_FUSION' in r('srt/environ.py'), 'proj-fusion-env'; \
assert 'dsa_indexer_projection_fusion_enabled' in r('srt/layers/attention/dsa/dsa_projection_fusion.py'), 'proj-fusion-helper'; \
assert 'use_dsa_indexer_projection_fusion' in r('srt/layers/attention/dsa/dsa_indexer.py'), 'proj-fusion-gate'; \
assert '_mqa_i32_safe_max_rows' in r('srt/layers/attention/dsa/dsa_indexer.py'), 'mqa-i32-chunk'; \
assert '_ck_csv_has_row' in r('srt/layers/quantization/fp8_utils.py'), 'ck-gfx942-gate'; \
assert 'SGLANG_ROCM_USE_CK_GEMM_GFX942' in r('srt/layers/quantization/fp8_utils.py'), 'ck-gfx942-env'; \
assert 'topk_packed_kernel' in r('kernels/jit/csrc/deepseek_v4/topk_v2.cuh'), 'topk2p-kernel'; \
assert 'SGLANG_DSA_USE_TOPK_V2_PACKED' in r('srt/environ.py'), 'topk2p-env'; \
assert '_gfx942_cu_count' in r('kernels/ops/attention/dsa/tilelang_kernel.py'), 'cu-count-fix'; \
assert 'kQRFp16CastScaleLog2' in r('kernels/aot/csrc/allreduce/quick_all_reduce.cuh'), 'qr-saturation-fix'; \
assert (R/'kernels/aot/include/hip/dsa_topk_coop.cuh').stat().st_size > 0, 'w2-coop-cuh'; \
assert 'coop' in r('kernels/aot/csrc/elementwise/topk.hip'), 'w2-topk-hip'; \
print('[OK] All production patch markers verified in source tree')"

# Layer 2b: rebuild sgl-kernel so fast_topk_v2 is the coop exact kernel.
# hipcc does not need a GPU. ti-builder 64-core, expect 30-60 min.
# Do not use `pip wheel .` here: pep517/setuptools auto-discovery treats
# csrc/cmake/include as top-level packages and aborts.
RUN set -eux; \
    cd /sgl-workspace/sglang/python/sglang/kernels/aot && \
    python3 setup_rocm.py build_ext --inplace && \
    so="$(find python/sgl_kernel . -name 'common_ops*.so' | head -1)"; \
    test -n "$so"; \
    dest="$(python3 -c 'import sgl_kernel, pathlib; print(pathlib.Path(sgl_kernel.__file__).parent)')"; \
    cp -v "$so" "$dest/"; \
    python3 -c "from sgl_kernel import fast_topk_v2; print('fast_topk_v2 OK')"

# ============================================================
# Layer 2c: aiter upgrade d9e5ef7 -> 4f613c2 (2026-09-19).
# Brings: opt dynmic_per_group_scaled_quant (#5631), indexer_qk_rope_quant
# rewrite (#5576), flydsl radix topk one-block (#5363), topk_select half
# formats (#5526), FMoE GEMM2 A addressing fix (#5519), triton 3.8 configs
# (#5612), cu80 rows in tuned_fmoe_glm5_1 (covers the 256/8 routed-only path
# left on default since 2026-08-23), and the flydsl fp8_mqa_logits BLOCK_KV=128
# + variant system + FNUZ fix that our Layer-3 overlays used to carry
# (upstreamed — overlays dropped).
# JIT .so modules rebuild on first use (~60-90s per kernel, once).
# ============================================================
RUN set -eux; \
    cd /sgl-workspace/aiter; \
    git fetch origin 4f613c281acd3fb9e47429a468458501ec966c15; \
    git checkout 4f613c281acd3fb9e47429a468458501ec966c15; \
    GPU_ARCHS=gfx942 pip install -e . --no-deps --no-build-isolation; \
    python3 -c "import importlib.metadata as md; print('aiter', md.version('amd-aiter'))"

# Upstream moved fp8_mqa_logits into kernels/mqa_logits/; sglang still imports
# the old flat path. Re-export until the sglang side moves.
COPY docker/rocm-mi308x-glm52-pd/patches/flydsl_mqa_compat.py /sgl-workspace/aiter/aiter/ops/flydsl/kernels/fp8_mqa_logits.py

# ============================================================
# Layer 4: AITER tuned GEMM/MoE configs for gfx942
# Overlay is a standalone overwrite (gate N=256 + indexer N=32 + N=160
# passthrough). Do not merge into the fat source CSV.
# ============================================================
COPY docker/rocm-mi308x-glm52/patches/glm5_1_gemm_gfx942.csv /sgl-workspace/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv
COPY docker/rocm-mi308x-glm52/patches/glm5_1_fmoe_gfx942.csv /sgl-workspace/aiter/aiter/configs/model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv
COPY docker/rocm-mi308x-glm52-pd/patches/mi308x_gfx942_bf16_gate_indexer.csv \
     /sgl-workspace/aiter/aiter/configs/model_configs/mi308x_gfx942_bf16_tuned_gemm.csv
COPY docker/rocm-mi308x-glm52-pd/scripts/gen_bf16_gate_indexer.py \
     /opt/aiter-scripts/gen_bf16_gate_indexer.py

# ============================================================
# Layer 5: libbnxt_re-rdmav34.so 238.1.138.5 (ts4 kernel bnxt_re ABI match)
# The base image ships libbnxt_re 235.2.86.0, which does NOT support the ts4
# kernel's bnxt_re ABI (kernel reports ABI 1, driver expects 7-8), so
# ibv_devinfo shows "No IB devices found". The 238.x driver matches the ts4
# kernel module ABI and is required for RDMA on ts4 nodes.
# ============================================================
COPY docker/rocm-mi308x-glm52-pd/patches/libbnxt_re/libbnxt_re-rdmav34.so /tmp/libbnxt_re-rdmav34.so
COPY docker/rocm-mi308x-glm52-pd/patches/libbnxt_re/libbnxt_re-235.2.86.0.so /tmp/libbnxt_re-235.2.86.0.so
RUN set -eux; \
    cp /tmp/libbnxt_re-rdmav34.so /usr/local/lib/libbnxt_re-rdmav34.so && \
    cp /tmp/libbnxt_re-rdmav34.so /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav34.so && \
    mkdir -p /usr/lib/x86_64-linux-gnu/libibverbs && \
    cp /tmp/libbnxt_re-235.2.86.0.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so && \
    ldconfig && \
    rm /tmp/libbnxt_re-rdmav34.so /tmp/libbnxt_re-235.2.86.0.so && \
    echo "[libbnxt_re] verbs=235.2.86.0 (kernel ABI v8) local=238.1.138.5" && \
    strings /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so | grep -m1 "235\."

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
ENV SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1
ENV SGLANG_PD_HOST_STAGING=0
ENV MC_DISABLE_HIP_TRANSPORT=1
ENV MOONCAKE_PROTOCOL=rdma

CMD ["/bin/bash"]
