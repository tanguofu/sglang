#!/usr/bin/env python3
"""Apply GDR patches to Mooncake source on a pod.
Patches:
  1. rdma_context.cpp — add HipDeviceGuard to fallback branch (ibv_reg_mr on GPU mem)
  2. worker_pool.cpp — add hipSetDevice(0) at transferWorker entry + cuda_alike.h include
  3. rdma_transport/CMakeLists.txt — propagate USE_HIP_DMABUF to rdma_transport target
"""
import sys

# Patch 1: rdma_context.cpp
rc_path = "/sgl-workspace/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp"
with open(rc_path) as f:
    src = f.read()
with open(rc_path + ".bak", "w") as f:
    f.write(src)

old = """    } else if (hipAttr.type == hipMemoryTypeDevice &&
               !isKernelDmabufSupported()) {
        // Kernel lacks CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY \u2014
        // ibv_reg_dmabuf_mr may succeed but transfers will silently fail.
        // Fail at registration time instead.
        mrMeta.addr = addr;
        mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);"""
new = """    } else if (hipAttr.type == hipMemoryTypeDevice &&
               !isKernelDmabufSupported()) {
        // Kernel lacks CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY.
        // Fall back to ibv_reg_mr on the GPU address directly (amdgpu
        // peermem via ib_register_peer_memory_client handles the DMA
        // mapping WITHOUT needing P2PDMA).
        // CRITICAL: must pin to the owning device so the peermem driver
        // uses the correct BAR. Without this, cross-node transfers fault
        // with "Memory access fault by GPU node-X" because the worker
        // thread's device context defaults to 0.
        HipDeviceGuard dev_guard(hipAttr.device);
        if (!dev_guard.set_ok()) {
            LOG(ERROR) << "Failed to set HIP device to " << hipAttr.device
                       << " for ibv_reg_mr fallback of " << (uintptr_t)addr;
            return ERR_CONTEXT;
        }
        mrMeta.addr = addr;
        mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);"""
if old in src:
    src = src.replace(old, new, 1)
    with open(rc_path, "w") as f:
        f.write(src)
    print("PATCH 1 (rdma_context.cpp): APPLIED")
elif "ibv_reg_mr fallback of" in src:
    print("PATCH 1 (rdma_context.cpp): SKIP (already patched)")
else:
    print("PATCH 1 (rdma_context.cpp): ERROR (pattern not found)")
    sys.exit(1)

# Patch 2: worker_pool.cpp
wp_path = "/sgl-workspace/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp"
with open(wp_path) as f:
    src = f.read()
with open(wp_path + ".bak", "w") as f:
    f.write(src)

if '#include "cuda_alike.h"' not in src:
    lines = src.split("\n")
    last_include_idx = 0
    for i, line in enumerate(lines[:50]):
        if line.startswith("#include"):
            last_include_idx = i
    lines.insert(last_include_idx + 1, '#include "cuda_alike.h"  // hipSetDevice (for GDR device context in worker threads)')
    src = "\n".join(lines)

old_tw = """void WorkerPool::transferWorker(int thread_id) {
    bindToSocket(numa_socket_id_);
    const static uint64_t kWaitPeriodInNano = 100000000;  // 100ms"""
new_tw = """void WorkerPool::transferWorker(int thread_id) {
    bindToSocket(numa_socket_id_);
    // CRITICAL: worker threads do not inherit the caller's HIP device
    // context. Without hipSetDevice, the default device (0) is used,
    // which causes "Memory access fault by GPU node-X" when posting
    // sends for slices on other GPUs. Set device 0 as a safe baseline;
    // registerMR's HipDeviceGuard pins the correct device per-buffer,
    // and ibv_post_send reuses the MR's lkey which already has the
    // correct BAR mapping.
    (void)hipSetDevice(0);
    const static uint64_t kWaitPeriodInNano = 100000000;  // 100ms"""
if old_tw in src:
    src = src.replace(old_tw, new_tw, 1)
    with open(wp_path, "w") as f:
        f.write(src)
    print("PATCH 2 (worker_pool.cpp): APPLIED")
elif "hipSetDevice(0)" in src:
    print("PATCH 2 (worker_pool.cpp): SKIP (already patched)")
else:
    print("PATCH 2 (worker_pool.cpp): ERROR (transferWorker pattern not found)")
    sys.exit(1)

# Patch 3: rdma_transport/CMakeLists.txt
cm_path = "/sgl-workspace/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/CMakeLists.txt"
with open(cm_path) as f:
    src = f.read()
with open(cm_path + ".bak", "w") as f:
    f.write(src)

if "USE_HIP_DMABUF" not in src:
    new_block = '''file(GLOB RDMA_SOURCES "*.cpp")

add_library(rdma_transport OBJECT ${RDMA_SOURCES})
target_link_libraries(rdma_transport PRIVATE JsonCpp::JsonCpp glog::glog
                                             pthread)

if(USE_MLX5DV)
  target_compile_definitions(rdma_transport PRIVATE USE_MLX5DV)
endif()

# Propagate HIP/HIP_DMABUF defines to rdma_transport so rdma_context.cpp
# picks the dmabuf/fallback GDR branch instead of the #else host-staging
# branch. Without this, USE_HIP_DMABUF is only set on transfer_engine
# (src/CMakeLists.txt:110) and rdma_context.cpp compiles the D2H/H2D
# bounce-buffer path, defeating GPU-Direct RDMA.
if(USE_HIP)
  target_compile_definitions(rdma_transport PRIVATE USE_HIP __HIP_PLATFORM_AMD__)
  if(USE_HIP_DMABUF)
    target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)
  endif()
endif()
'''
    with open(cm_path, "w") as f:
        f.write(new_block)
    print("PATCH 3 (CMakeLists.txt): APPLIED")
else:
    print("PATCH 3 (CMakeLists.txt): SKIP (already has USE_HIP_DMABUF)")

print("All patches applied successfully.")
