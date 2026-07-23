#!/bin/bash
# Fix Mooncake to enable GPU-direct RDMA (GDR) on AMD MI308X + bnxt_re.
#
# Root cause: USE_HIP_DMABUF is defined on transfer_engine target (PRIVATE),
# but rdma_context.cpp belongs to rdma_transport OBJECT library which doesn't
# inherit PRIVATE definitions. So rdma_context.cpp compiles WITHOUT
# USE_HIP_DMABUF, falling into the #else host-staging branch.
#
# Fix: add target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)
# so rdma_context.cpp compiles with the dmabuf/GDR code path.
#
# When isKernelDmabufSupported() returns false (kernel lacks
# CONFIG_PCI_P2PDMA), the USE_HIP_DMABUF branch falls back to
# ibv_reg_mr(pd, gpu_ptr, ...) — which we verified works on MI308X +
# bnxt_re via end-to-end RDMA write test (data integrity confirmed).
#
# This eliminates the 43.2 GiB per-rank hipMemcpy bottleneck caused by
# the old host-staging workaround.
set -eux
cd /sgl-workspace/Mooncake

# Patch: add USE_HIP_DMABUF to rdma_transport target
python3 -c "
f = 'mooncake-transfer-engine/src/transport/rdma_transport/CMakeLists.txt'
s = open(f).read()
old = 'add_library(rdma_transport OBJECT \${RDMA_SOURCES})'
new = '''add_library(rdma_transport OBJECT \${RDMA_SOURCES})

# Propagate USE_HIP_DMABUF to rdma_transport so rdma_context.cpp
# compiles the GPU-direct RDMA (dmabuf/ibv_reg_mr) code path.
# Without this, USE_HIP_DMABUF is only defined on transfer_engine
# (PRIVATE) and rdma_context.cpp falls into the #else host-staging branch.
if(USE_HIP AND USE_HIP_DMABUF)
  target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)
endif()'''
if old in s and 'rdma_transport PRIVATE USE_HIP_DMABUF' not in s:
    s = s.replace(old, new, 1)
    open(f, 'w').write(s)
    print('CMakeLists.txt patched: USE_HIP_DMABUF propagated to rdma_transport')
else:
    print('CMakeLists.txt already patched or pattern not found')
"

# Also patch the transport-level CMakeLists.txt in case USE_HIP_DMABUF
# needs to be visible to other transport files (e.g., hip_transport)
python3 -c "
f = 'mooncake-transfer-engine/src/transport/CMakeLists.txt'
import os
if not os.path.exists(f):
    print('transport/CMakeLists.txt does not exist, skipping')
    exit(0)
s = open(f).read()
if 'USE_HIP_DMABUF' not in s and 'rdma_transport' in s:
    # Add USE_HIP_DMABUF propagation if not present
    print('transport/CMakeLists.txt: no changes needed (rdma_transport handles it)')
else:
    print('transport/CMakeLists.txt: already has USE_HIP_DMABUF or no rdma_transport')
"
