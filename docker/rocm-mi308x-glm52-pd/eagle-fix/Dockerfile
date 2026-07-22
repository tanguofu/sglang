# GLM-5.2-FP8 PD worker for AMD MI308X (gfx942) — EAGLE+DSA fixed image.
#
# Base: sglang-glm52-308x:fix-eagle-coredump-v3 (has working EAGLE + dsa attention
# backend on GLM-5.2 — quality verified 10/10 on prof19/test32 pods).
#
# This image ONLY overlays:
#   1. Mooncake engine.so with USE_HIP_DMABUF=ON (GDR via HIP dmabuf path,
#      respects MC_DISABLE_HIP_TRANSPORT=1). Built from post1-0716-gdr-v3 image
#      which has 14 dmabuf refs + the CMake USE_HIP_DMABUF propagation fix.
#   2. libbnxt_re-rdmav34.so 238.1.138.5 (ts4 kernel bnxt_re ABI match;
#      base image ships 235.x which fails on ts4 nodes).
#
# The base image already has the EAGLE+DSA fix (newer DSA backend, 2765 lines vs
# post1-0716's 2567 lines). We do NOT apply any SGLang patches — the base's SGLang
# code is newer and works correctly with EAGLE + dsa backend.
#
# Build (from repo root):
#   docker build -f docker/rocm-mi308x-glm52-pd/eagle-fix/Dockerfile \
#     -t sglang-glm52-308x-pd:eagle-gdr-v1 .

FROM mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3

LABEL maintainer="guofutan"
LABEL base-image="sglang-glm52-308x:fix-eagle-coredump-v3"
LABEL rdma-mode="GDR (HIP dmabuf, no host staging)"
LABEL eagle-dsa="fixed (base image has working EAGLE + dsa backend)"

# ============================================================
# Layer 1: GDR-enabled Mooncake engine.so
# Replaces the base image's engine.so (which has 0 dmabuf refs — no GDR)
# with the GDR-capable one built from post1-0716-gdr-v3 (14 dmabuf refs,
# USE_HIP_DMABUF=ON, respects MC_DISABLE_HIP_TRANSPORT=1).
# ============================================================
COPY docker/rocm-mi308x-glm52-pd/eagle-fix/engine.cpython-310-x86_64-linux-gnu.so /tmp/gdr-engine.so
RUN set -eux; \
    cp /tmp/gdr-engine.so /opt/venv/lib/python3.10/site-packages/mooncake/engine.cpython-310-x86_64-linux-gnu.so && \
    ldconfig && \
    rm /tmp/gdr-engine.so && \
    DMABUF_REFS=$(strings /opt/venv/lib/python3.10/site-packages/mooncake/engine.cpython-310-x86_64-linux-gnu.so | grep -c dmabuf); \
    if [ "$DMABUF_REFS" -lt 5 ]; then \
        echo "ERROR: engine.so has only $DMABUF_REFS dmabuf refs (expected >=5). GDR is NOT compiled in."; \
        exit 1; \
    fi; \
    echo "[Mooncake] VERIFIED: engine.so has $DMABUF_REFS dmabuf refs (GDR compiled in)"

# ============================================================
# Layer 2: libbnxt_re 238.1.138.5 (ts4 kernel ABI match)
# Base image ships 235.2.86.0 which fails on ts4 nodes (bnxt_re ABI mismatch).
# ============================================================
COPY docker/rocm-mi308x-glm52-pd/eagle-fix/libbnxt_re-rdmav34.so /tmp/libbnxt_re-rdmav34.so
RUN set -eux; \
    cp /tmp/libbnxt_re-rdmav34.so /usr/local/lib/libbnxt_re-rdmav34.so && \
    cp /tmp/libbnxt_re-rdmav34.so /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav34.so && \
    ldconfig && \
    rm /tmp/libbnxt_re-rdmav34.so && \
    echo "[libbnxt_re] installed 238.1.138.5 (ts4 kernel ABI match)" && \
    strings /usr/local/lib/libbnxt_re-rdmav34.so | grep -m1 "238\."

CMD ["/bin/bash"]
