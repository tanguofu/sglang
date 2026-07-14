# Dockerfile for vLLM PD + LMCache on AMD MI308X (gfx942, ROCm 7.2)
# Branch: 308x-vllm-llcache-1pd
#
# KV cache transfer: NIXL/UCX RDMA (bnxt_re RoCE v2, host pinned memory, no peer_mem needed)
#
# Build on ti-builder:
#   docker build -f docker/rocm-mi308x-vllm-pd/Dockerfile \
#     -t mirrors.tencent.com/ti-platform/vllm-glm52-308x-pd:latest .

# -- Stage 1: Base vLLM ROCm image --
FROM vllm/vllm-openai-rocm:v0.25.1 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTORCH_ROCM_ARCH=gfx942 \
    HSA_ENABLE_SDMA=0 \
    HIP_FORCE_DEV_KERNARG=1 \
    HSA_NO_SCRATCH_RECLAIM=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NCCL_DEBUG=INFO \
    NCCL_IB_DISABLE=0 \
    NCCL_IB_HCA=bnxt_re \
    NCCL_NET_GDR_LEVEL=0 \
    NCCL_SOCKET_IFNAME=bond0

# -- Install RDMA dependencies (UCX already built in base vLLM ROCm image) --
RUN apt-get update && apt-get install -y --no-install-recommends \
    rdma-core \
    ibverbs-utils \
    perftest \
    && rm -rf /var/lib/apt/lists/*

# -- Remove CUDA-only packages that conflict with ROCm --
RUN pip uninstall -y nixl nixl-cu12 cupy-cuda12x cufile-python cuda-pathfinder 2>/dev/null || true

# -- Clone and build LMCache from source (HIP mode, not CUDA) --
RUN git clone --depth 1 https://github.com/LMCache/LMCache.git /opt/LMCache && \
    cd /opt/LMCache && \
    BUILD_WITH_HIP=1 pip install -e . --no-build-isolation && \
    python3 -c "import lmcache; print('LMCache installed OK')"

# -- Install disagg proxy server for PD routing --
RUN pip install --no-cache-dir fastapi uvicorn httpx

# -- Verify vLLM + PyTorch ROCm --
RUN python3 -c "import vllm; print('vLLM version:', vllm.__version__)" && \
    python3 -c "import torch; print('PyTorch ROCM:', torch.version.hip)"

# -- Copy entrypoint + proxy server --
COPY docker/rocm-mi308x-vllm-pd/start_server.sh /start_server.sh
COPY docker/rocm-mi308x-vllm-pd/disagg_proxy.py /disagg_proxy.py
COPY docker/rocm-mi308x-vllm-pd/disagg_proxy_server.py /disagg_proxy_server.py
RUN chmod +x /start_server.sh

ENTRYPOINT ["/start_server.sh"]
