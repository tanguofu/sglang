# DeepSeek-V4.1-Flash on MI308X (gfx942) — mHC #1 launch fusion.
#
# PURE PREPARATION: this Dockerfile is NOT built in this round.  Do not build,
# push, deploy or restart anything without an explicit GO + a frozen window.
# The live service is ds41-flash.jmpti.woa.com (m20, 1P1D, restart = 断服).
#
# Image delta = exactly two .py files (nothing else, no .so / no AOT rebuild):
#   srt/models/deepseek_v4_mhc.py       (env-gated call site; OFF arm verbatim)
#   kernels/ops/layernorm/mhc.py        (+ fused kernel/wrapper/gate; shipped
#                                        kernels byte-identical)
# patch_mhc_fusion.py asserts the pulled m20 base is an insertion-only ancestor
# of the reviewed worktree files and emits the REVIEWED bytes verbatim (the ones
# the GPU torch.equal was measured against).  check_mhc_fusion.py additionally
# PINNS the m20 base md5 (read from the pinned digest on ti-builder, 2026-11-08,
# and cross-checked against the live m20 pod + fork HEAD d7dab9171c).
#
# M20_BASE is pinned to the m20 image DIGEST, not the tag: registry tags have
# been re-pushed before (see docker/ds41-image/m20/Dockerfile:33).  This is the
# m20 RepoDigest read from ti-builder dind-builder4; build.sh overrides it with
# --build-arg M20_BASE=$BASE (same digest via M20_DIGEST).
ARG M20_BASE=mirrors.tencent.com/ti-platform/sglang-glm52-308x@sha256:5a0fcb8a2e12eafd6dab9f8ea632051a27152e63e342229fa506484c8d934374
FROM ${M20_BASE}

COPY sglang/ /sgl-workspace/sglang/python/sglang/
COPY baseline/ /tmp/mhc-fusion-baseline/
COPY check_mhc_fusion.py /tmp/check_mhc_fusion.py
COPY verify_mhc_fusion.py /tmp/verify_mhc_fusion.py

RUN set -eu; \
    P=/sgl-workspace/sglang/python/sglang; \
    M=$P/srt/models/deepseek_v4_mhc.py; \
    H=$P/kernels/ops/layernorm/mhc.py; \
    present() { grep -qF "$2" "$1" || { echo "mhc-fusion: MISSING $2 in $(basename "$1")" >&2; exit 1; }; \
      echo "mhc-fusion: present $2 in $(basename "$1")"; }; \
    present "$M" 'SGLANG_DS41_MHC_FUSED_SINKHORN'; \
    present "$M" 'hc_split_sinkhorn_fused('; \
    present "$H" 'ENV_MHC_FUSED_SINKHORN'; \
    present "$H" 'def hc_split_sinkhorn_fused_kernel'; \
    present "$H" 'mixes_shared[j] = mixes_shared[j] * rs'; \
    export PYTHONDONTWRITEBYTECODE=1; \
    python3 /tmp/check_mhc_fusion.py \
      "$M" /tmp/mhc-fusion-baseline/deepseek_v4_mhc.py \
      "$H" /tmp/mhc-fusion-baseline/mhc.py; \
    python3 /tmp/verify_mhc_fusion.py "$H" "$M"; \
    rm -rf /tmp/mhc-fusion-baseline /tmp/check_mhc_fusion.py /tmp/verify_mhc_fusion.py; \
    find /sgl-workspace/sglang/python/sglang -name '__pycache__' -prune -exec rm -rf {} + ; \
    echo "mhc-fusion: build-time check + verify done, scratch removed"

# GPU-free build-time proof (AST + fp32 scalar) above.  It NEVER claims a
# torch.equal was run.  Bit-exactness by construction + static exhibit only;
# GPU torch.equal remains a rollout gate (verify/verify_mhc_fusion.py).
