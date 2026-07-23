#!/bin/bash
# Apply GDR (GPUDirect RDMA via HIP dmabuf) patches to Mooncake.
#
# Goal: enable the USE_HIP_DMABUF path in rdma_context.cpp so that GPU memory
# is registered via hsa_amd_portable_export_dmabuf + ibv_reg_dmabuf_mr (true
# GPUDirect RDMA), NOT host staging (hipMallocHost + hipMemcpy + ibv_reg_mr).
#
# Changes:
#   1. transfer_engine_impl.cpp: respect MC_DISABLE_HIP_TRANSPORT=1 env var
#      (disables intra-node HIP IPC transport; required for cross-node PD).
#   2. rdma_context.cpp: ensure the #else fallback does NOT do host staging.
#      The base image's Mooncake source may have been patched with host staging
#      by a prior build layer — we strip any staging code so the #else block
#      only does plain ibv_reg_mr on host memory, and GPU memory flows through
#      the USE_HIP_DMABUF path (compiled in via CMakeCache USE_HIP_DMABUF=ON).
#
# Idempotent. Run from /sgl-workspace/Mooncake.
set -eux

# ------------------------------------------------------------------
# Patch 1: MC_DISABLE_HIP_TRANSPORT in transfer_engine_impl.cpp
# ------------------------------------------------------------------
python3 -c "
f = 'mooncake-transfer-engine/src/transfer_engine_impl.cpp'
s = open(f).read()
if 'MC_DISABLE_HIP_TRANSPORT' in s:
    print('transfer_engine_impl.cpp: MC_DISABLE_HIP_TRANSPORT already present')
else:
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
        print('transfer_engine_impl.cpp: patched MC_DISABLE_HIP_TRANSPORT')
    else:
        print('transfer_engine_impl.cpp: WARN pattern not found, skipping')
"

# ------------------------------------------------------------------
# Patch 2: Strip host-staging code from rdma_context.cpp #else block.
# The GDR path is in the #elif defined(USE_HIP_DMABUF) block and requires
# no patching — it is compiled in when CMakeCache has USE_HIP_DMABUF=ON.
# We only need to ensure the #else block does NOT inject host staging,
# so that GPU pointers (hipMemoryTypeDevice) fall through to the USE_HIP_DMABUF
# branch above #else.
# ------------------------------------------------------------------
python3 -c "
f = 'mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp'
s = open(f).read()

# Detect and strip host-staging injection in the #else block.
# The host-staging patch inserts a hipMallocHost+hipMemcpy+ibv_reg_mr block
# inside #else. We replace it back to the original plain ibv_reg_mr.
staging_markers = ['staging_host_buf', 'staging_gpu_addr', 'RDMA host staging: GPU=']
if not any(m in s for m in staging_markers):
    print('rdma_context.cpp: no host-staging code found (GDR path intact)')
else:
    # Find the #else block that was patched and restore plain ibv_reg_mr.
    # The patched #else block looks like:
    #   #else
    #   {
    #       hipPointerAttribute_t elseAttr{};
    #       ... hipMallocHost ... hipMemcpy ... ibv_reg_mr(pd_, host_buf, ...) ...
    #   }
    #   #endif
    # We replace it with:
    #   #else
    #       mrMeta.addr = addr;
    #       mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);
    #   #endif
    import re
    # Match the #else { ... } #endif block that contains staging code
    pat = re.compile(
        r'#else\s*\n\s*\{[^#]*?staging_host_buf[^#]*?\}\s*\n#endif',
        re.DOTALL
    )
    m = pat.search(s)
    if m:
        s = s[:m.start()] + (
            '#else\n'
            '    mrMeta.addr = addr;\n'
            '    mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);\n'
            '#endif'
        ) + s[m.end():]
        open(f, 'w').write(s)
        print('rdma_context.cpp: stripped host-staging code from #else block (GDR restored)')
    else:
        # Try a more lenient pattern
        print('rdma_context.cpp: WARN staging markers found but regex did not match; manual check needed')
        # Print the #else region for debugging
        idx = s.find('#else')
        if idx >= 0:
            print('--- #else region (first 800 chars) ---')
            print(s[idx:idx+800])
"

# ------------------------------------------------------------------
# Patch 3: Strip staging fields from rdma_context.h MemoryRegionMeta.
# The host-staging patch adds staging_host_buf / staging_gpu_addr / staging_length
# fields to the struct. They are unused by the GDR path but harmless; we strip
# them for cleanliness and to avoid confusion.
# ------------------------------------------------------------------
python3 -c "
f = 'mooncake-transfer-engine/include/transport/rdma_transport/rdma_context.h'
s = open(f).read()
if 'staging_host_buf' not in s:
    print('rdma_context.h: no staging fields (clean)')
else:
    old = '''struct ibv_mr *mr;
    void *staging_host_buf = nullptr;
    void *staging_gpu_addr = nullptr;
    size_t staging_length = 0;
};'''
    new = '''struct ibv_mr *mr;
};'''
    if old in s:
        s = s.replace(old, new, 1)
        open(f, 'w').write(s)
        print('rdma_context.h: stripped staging fields from MemoryRegionMeta')
    else:
        print('rdma_context.h: WARN staging fields present but pattern not matched')
"

# ------------------------------------------------------------------
# Patch 4: Fix CMake USE_HIP_DMABUF compile definition propagation.
#
# Root cause: Mooncake's CMakeLists.txt adds the USE_HIP_DMABUF compile
# definition to the `transfer_engine` target (PRIVATE), but rdma_context.cpp
# (which contains the #if defined(USE_HIP_DMABUF) blocks) is compiled as part
# of the `rdma_transport` OBJECT library, which is a separate target. PRIVATE
# definitions do not propagate to object-library dependencies, so rdma_context.cpp
# gets compiled WITHOUT -DUSE_HIP_DMABUF, causing the #elif defined(USE_HIP_DMABUF)
# GDR path to be skipped and falling through to #else (plain ibv_reg_mr on host
# memory, which fails for GPU pointers).
#
# Fix: add `target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)`
# and link hsa-runtime64 to rdma_transport, mirroring what the existing CMake
# block does for transfer_engine.
#
# Without this patch, engine.so has 0 dmabuf refs and GDR is silently disabled
# even though CMakeCache shows USE_HIP_DMABUF:BOOL=ON.
# ------------------------------------------------------------------
python3 -c "
import os
f = 'mooncake-transfer-engine/src/transport/rdma_transport/CMakeLists.txt'
s = open(f).read()
if 'USE_HIP_DMABUF' in s:
    print('rdma_transport/CMakeLists.txt: USE_HIP_DMABUF already present')
else:
    # Append the dmabuf compile definition + hsa-runtime64 link after the
    # existing target_link_libraries line. This mirrors the transfer_engine
    # block in src/CMakeLists.txt lines 106-118.
    addition = '''

# GDR: enable HIP dmabuf MR registration path in rdma_context.cpp.
# Mirrors the transfer_engine block in src/CMakeLists.txt. Without this,
# the #if defined(USE_HIP_DMABUF) blocks in rdma_context.cpp are compiled
# out because PRIVATE definitions do not propagate to OBJECT libraries.
if(USE_HIP_DMABUF)
  find_package(hsa-runtime64 CONFIG)
  if(hsa-runtime64_FOUND)
    target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)
    target_link_libraries(rdma_transport PRIVATE hsa-runtime64::hsa-runtime64)
    message(STATUS \"HIP dmabuf MR registration enabled for rdma_transport (hsa-runtime64 found)\")
  else()
    message(STATUS \"HIP dmabuf MR registration disabled for rdma_transport (hsa-runtime64 not found)\")
  endif()
endif()
'''
    s = s.rstrip() + '\n' + addition
    open(f, 'w').write(s)
    print('rdma_transport/CMakeLists.txt: added USE_HIP_DMABUF compile definition + hsa-runtime64 link')
"

# Force CMake reconfigure on next make by touching the top-level CMakeLists.
touch mooncake-transfer-engine/src/transport/rdma_transport/CMakeLists.txt

echo "[Mooncake] GDR patches applied (USE_HIP_DMABUF path, no host staging)"
