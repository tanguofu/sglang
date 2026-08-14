# GLM-5.2-FP8 MI308X worker: latest SGLang + Aiter + Mooncake from source.
# Toolchain base is official ROCm 7.2 sgl-dev (compiler / PyTorch / TileLang only).
# Bundled sglang/aiter/mooncake in the base image are replaced.
#
# Build on ti-builder (repo root):
#   docker build -f docker/rocm-mi308x-glm52-pd/Dockerfile \
#     --build-arg AITER_COMMIT=... \
#     --build-arg MOONCAKE_COMMIT=... \
#     -t mirrors.tencent.com/ti-platform/sglang-glm52-308x:pd-l3-20260814 \
#     .
#
# Do NOT set PREBUILD_KERNELS=1 (needs rocminfo / GPU).
# Guangzhou 5.4 must use libbnxt_re-235.2.86.0 (copy onto build context first).

ARG BASE_IMAGE=mirrors.tencent.com/ti-platform/sgl-dev:v0.5.17-rocm720-mi30x-20260809
FROM ${BASE_IMAGE}

ARG SGLANG_SHA=fe0c18effd21
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_COMMIT=1b741c0e3ee2278e123a395fb23767ddbf8f8800
ARG MOONCAKE_REPO=https://github.com/kvcache-ai/Mooncake.git
ARG MOONCAKE_COMMIT=91c1c455cf275a1231a20319c196cb4f46b9cb5f
ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.5.18.dev0+fe0c18effd
ARG GPU_ARCH_LIST=gfx942

LABEL maintainer="guofutan"
LABEL gpu-arch="gfx942"
LABEL sglang.sha="${SGLANG_SHA}"
LABEL aiter.sha="${AITER_COMMIT}"
LABEL mooncake.sha="${MOONCAKE_COMMIT}"
LABEL base.image="sgl-dev:v0.5.17-rocm720-mi30x-20260809"
LABEL rdma-mode="peermem + ibv_reg_mr (Guangzhou 5.4, no HCA dmabuf)"

USER root
WORKDIR /sgl-workspace

ENV PYTORCH_ROCM_ARCH=${GPU_ARCH_LIST}
ENV GPU_ARCHS=${GPU_ARCH_LIST}
ENV AITER_USE_SYSTEM_TRITON=1
ENV PREBUILD_KERNELS=0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}

# ------------------------------------------------------------
# 1) Mooncake @ pinned SHA
# ------------------------------------------------------------
RUN set -eux; \
    rm -rf /sgl-workspace/Mooncake; \
    git clone --recurse-submodules "${MOONCAKE_REPO}" /sgl-workspace/Mooncake; \
    git -C /sgl-workspace/Mooncake checkout -f "${MOONCAKE_COMMIT}"; \
    git -C /sgl-workspace/Mooncake submodule update --init --recursive

COPY docker/rocm-mi308x-glm52-pd/patches/mooncake/apply_host_staging.sh /tmp/apply_mooncake.sh
RUN set -eux; \
    chmod +x /tmp/apply_mooncake.sh; \
    /tmp/apply_mooncake.sh || echo "[Mooncake] host-staging patch skipped (patterns may already be upstream)"; \
    cd /sgl-workspace/Mooncake; \
    mkdir -p build && cd build; \
    cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON; \
    make -j "$(nproc)"; \
    make install; \
    so=$(find /sgl-workspace/Mooncake/build -name 'engine.cpython-*-x86_64-linux-gnu.so' | head -1); \
    test -n "$so"; \
    dest=$(python3 -c "import mooncake, pathlib; print(pathlib.Path(mooncake.__file__).parent)"); \
    cp "$so" "$dest/$(basename "$so")"; \
    ldconfig; \
    echo "[Mooncake] installed $(basename "$so") @ ${MOONCAKE_COMMIT}"

# ------------------------------------------------------------
# 2) Aiter @ pinned SHA (JIT at runtime, no PREBUILD)
# ------------------------------------------------------------
RUN set -eux; \
    pip uninstall -y aiter || true; \
    rm -rf /sgl-workspace/aiter; \
    git clone --recurse-submodules "${AITER_REPO}" /sgl-workspace/aiter; \
    git -C /sgl-workspace/aiter checkout -f "${AITER_COMMIT}"; \
    git -C /sgl-workspace/aiter submodule update --init --recursive; \
    cd /sgl-workspace/aiter; \
    pip install -r requirements.txt; \
    GPU_ARCHS="${GPU_ARCH_LIST}" PREBUILD_KERNELS=0 \
      pip install --config-settings editable_mode=compat -e .; \
    echo "export PYTHONPATH=/sgl-workspace/aiter:\${PYTHONPATH}" >> /etc/bash.bashrc

# gfx942 tuned configs as extra model_configs files (never overwrite source CSVs)
COPY docker/rocm-mi308x-glm52/patches/glm5_1_gemm_gfx942.csv \
     /sgl-workspace/aiter/aiter/configs/model_configs/mi308x_gfx942_bf16_tuned_gemm.csv
COPY docker/rocm-mi308x-glm52/patches/glm5_1_fmoe_gfx942.csv \
     /sgl-workspace/aiter/aiter/configs/model_configs/mi308x_gfx942_fmoe.csv

# ------------------------------------------------------------
# 3) SGLang = this repo (upstream/main + local docker/)
# ------------------------------------------------------------
RUN pip uninstall -y sglang sgl_kernel || true
COPY python /sgl-workspace/sglang/python
COPY LICENSE /sgl-workspace/sglang/LICENSE
COPY README.md /sgl-workspace/sglang/README.md

RUN set -eux; \
    cd /sgl-workspace/sglang/python/sglang/kernels/aot; \
    rm -f pyproject.toml && mv pyproject_rocm.toml pyproject.toml; \
    AMDGPU_TARGET="${GPU_ARCH_LIST}" python setup_rocm.py install; \
    cd /sgl-workspace/sglang; \
    rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml; \
    export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION}"; \
    python -m pip --no-cache-dir install --no-deps -e "python[srt_hip]"; \
    python -c "import sglang, pathlib; print('sglang', pathlib.Path(sglang.__file__).resolve())"

# ------------------------------------------------------------
# 4) Guangzhou 5.4 bnxt_re userspace (ABI 7-8). Never bake 238.x here.
# ------------------------------------------------------------
COPY docker/rocm-mi308x-glm52-pd/patches/libbnxt_re/libbnxt_re-235.2.86.0.so /tmp/libbnxt_re-235.2.86.0.so
RUN set -eux; \
    mkdir -p /usr/lib/x86_64-linux-gnu/libibverbs /usr/local/lib/x86_64-linux-gnu; \
    cp /tmp/libbnxt_re-235.2.86.0.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so; \
    cp /tmp/libbnxt_re-235.2.86.0.so /usr/local/lib/libbnxt_re-rdmav34.so; \
    ldconfig; \
    strings /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so | grep -m1 '235\.'; \
    rm -f /tmp/libbnxt_re-235.2.86.0.so

# ------------------------------------------------------------
# 5) Kernel unit tests (also used by k8s Jobs)
# ------------------------------------------------------------
COPY docker/rocm-mi308x-glm52-pd/kernel-tests /opt/kernel-tests

ENV HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV NCCL_DEBUG=WARN
ENV HSA_ENABLE_SDMA=0
ENV HIP_FORCE_DEV_KERNARG=1
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV NCCL_CUMEM_ENABLE=0
ENV NCCL_NVLS_ENABLE=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
ENV SGLANG_DISABLE_CUDNN_CHECK=1
ENV SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM=1
ENV SGLANG_USE_AITER=1
ENV SGLANG_SET_CPU_AFFINITY=1

CMD ["/bin/bash"]
