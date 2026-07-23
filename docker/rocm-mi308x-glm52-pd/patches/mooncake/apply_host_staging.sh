#!/bin/bash
# Apply Mooncake C++ patches for bnxt_re + MI308X GPU-direct RDMA.
#
# HISTORY: This script previously added host-staging patches (Patch 1 + Patch 2)
# as a workaround for GPU-direct RDMA not working. We have since verified that
# GPU-direct RDMA (ibv_reg_mr on GPU memory + RDMA write) DOES work on MI308X +
# bnxt_re (end-to-end test confirmed data integrity). The fix is now in
# enable_gdr.sh which propagates USE_HIP_DMABUF to the rdma_transport target.
#
# This script now only applies Patch 3 (MC_DISABLE_HIP_TRANSPORT env var support)
# which disables intra-node HIP transport for cross-node PD scenarios.
set -eux
cd /sgl-workspace/Mooncake

# Patch 1: Add staging fields to MemoryRegionMeta in rdma_context.h
# KEPT for backward compatibility — the USE_HIP_DMABUF path doesn't use these
# fields, but they're harmless and keep the struct definition consistent.
python3 -c "
f = 'mooncake-transfer-engine/include/transport/rdma_transport/rdma_context.h'
s = open(f).read()
old = 'struct ibv_mr *mr;\n};'
new = '''struct ibv_mr *mr;
    void *staging_host_buf = nullptr;
    void *staging_gpu_addr = nullptr;
    size_t staging_length = 0;
};'''
if old in s:
    s = s.replace(old, new, 1)
    open(f, 'w').write(s)
    print('Header patched: staging fields added')
else:
    print('Header already patched or pattern not found')
"

# Patch 2: REMOVED — host staging in #else block no longer needed.
# The enable_gdr.sh script ensures USE_HIP_DMABUF is defined on rdma_transport,
# so rdma_context.cpp compiles the dmabuf/GDR path (not the #else host-staging path).
# When isKernelDmabufSupported() returns false (no CONFIG_PCI_P2PDMA), the GDR
# path falls back to ibv_reg_mr(pd, gpu_ptr, ...) which we verified works.

# Patch 3: Add MC_DISABLE_HIP_TRANSPORT to transfer_engine_impl.cpp
# This disables intra-node HIP transport (IPC) which is not needed for
# cross-node PD and can cause issues on MI308X.
python3 -c "
f = 'mooncake-transfer-engine/src/transfer_engine_impl.cpp'
s = open(f).read()
old = '''        {
            Transport* hip_transport =
                multi_transports_->installTransport(\"hip\", nullptr);
            if (!hip_transport) {
                LOG(WARNING) << \"Failed to install HIP transport \"
                                \"(intra-node GPU P2P unavailable)\";
            } else {
                LOG(INFO) << \"HIP transport installed for intra-node GPU P2P\";
            }
        }'''
new = '''        {
            if (!std::getenv(\"MC_DISABLE_HIP_TRANSPORT\") ||
                std::string(std::getenv(\"MC_DISABLE_HIP_TRANSPORT\")) != \"1\") {
                Transport* hip_transport =
                    multi_transports_->installTransport(\"hip\", nullptr);
                if (!hip_transport) {
                    LOG(WARNING) << \"Failed to install HIP transport \"
                                    \"(intra-node GPU P2P unavailable)\";
                } else {
                    LOG(INFO) << \"HIP transport installed for intra-node GPU P2P\";
                }
            } else {
                LOG(INFO) << \"HIP transport disabled by MC_DISABLE_HIP_TRANSPORT=1\";
            }
        }'''
if old in s:
    s = s.replace(old, new, 1)
    open(f, 'w').write(s)
    print('transfer_engine_impl.cpp patched: MC_DISABLE_HIP_TRANSPORT')
else:
    print('transfer_engine_impl.cpp already patched or pattern not found')
"
