#!/usr/bin/env python3
"""Patch Mooncake for cross-node GDR via amdgpu peermem (no P2PDMA).

1. rdma_context.cpp fallback ibv_reg_mr(GPU): HipDeviceGuard so peermem
   uses the owning GPU BAR (fixes Memory access fault by GPU node-X).
2. worker_pool.cpp transferWorker: hipSetDevice(0) so worker threads have
   a valid HIP context.

Idempotent. Does not enable host-staging bounce buffers.
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/Mooncake")
CTX = ROOT / "mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp"
WORKER = ROOT / "mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp"

OLD_FALLBACK = """    } else if (hipAttr.type == hipMemoryTypeDevice &&
               !isKernelDmabufSupported()) {
        // Kernel lacks CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY —
        // ibv_reg_dmabuf_mr may succeed but transfers will silently fail.
        // Fail at registration time instead.
        mrMeta.addr = addr;
        mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);
"""

NEW_FALLBACK = """    } else if (hipAttr.type == hipMemoryTypeDevice &&
               !isKernelDmabufSupported()) {
        // Kernel lacks CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY.
        // ibv_reg_mr on the GPU address: amdgpu peermem via
        // ib_register_peer_memory_client (no P2PDMA needed).
        // FIX(gdr-peermem-device-guard): pin owning device so peermem
        // uses the correct BAR. Worker threads default to device 0.
        HipDeviceGuard dev_guard(hipAttr.device);
        if (!dev_guard.set_ok()) {
            LOG(ERROR) << "Failed to set HIP device to " << hipAttr.device
                       << " for ibv_reg_mr fallback of " << (uintptr_t)addr;
            return ERR_CONTEXT;
        }
        mrMeta.addr = addr;
        mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);
"""

OLD_WORKER = """void WorkerPool::transferWorker(int thread_id) {
    bindToSocket(numa_socket_id_);
"""

NEW_WORKER = """void WorkerPool::transferWorker(int thread_id) {
    bindToSocket(numa_socket_id_);
    // FIX(gdr-peermem-worker-device): workers do not inherit caller HIP
    // context. Device 0 is a safe baseline; registerMR HipDeviceGuard
    // pins the correct BAR per buffer.
    (void)hipSetDevice(0);
"""

changed = []

ctx = CTX.read_text()
if "FIX(gdr-peermem-device-guard)" in ctx:
    changed.append("rdma_context: already patched")
elif OLD_FALLBACK not in ctx:
    raise SystemExit(f"rdma_context.cpp fallback anchor not found: {CTX}")
else:
    CTX.write_text(ctx.replace(OLD_FALLBACK, NEW_FALLBACK, 1))
    changed.append("rdma_context: HipDeviceGuard applied")

wrk = WORKER.read_text()
if "FIX(gdr-peermem-worker-device)" in wrk:
    changed.append("worker_pool: already patched")
elif OLD_WORKER not in wrk:
    raise SystemExit(f"worker_pool.cpp transferWorker anchor not found: {WORKER}")
else:
    if "#include <hip/hip_runtime.h>" not in wrk:
        wrk = wrk.replace(
            '#include "transport/rdma_transport/rdma_transport.h"\n',
            '#include "transport/rdma_transport/rdma_transport.h"\n'
            "#include <hip/hip_runtime.h>\n",
            1,
        )
    WORKER.write_text(wrk.replace(OLD_WORKER, NEW_WORKER, 1))
    changed.append("worker_pool: hipSetDevice applied")

cmake = (
    ROOT
    / "mooncake-transfer-engine/src/transport/rdma_transport/CMakeLists.txt"
)
if cmake.exists() and "USE_HIP_DMABUF" not in cmake.read_text():
    cmake.write_text(
        cmake.read_text()
        + """
if(USE_HIP)
  target_compile_definitions(rdma_transport PRIVATE USE_HIP)
  if(USE_HIP_DMABUF)
    target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)
  endif()
endif()
"""
    )
    changed.append("CMakeLists: USE_HIP_DMABUF on rdma_transport")

print("[ok] apply_gdr_peermem: " + "; ".join(changed))
