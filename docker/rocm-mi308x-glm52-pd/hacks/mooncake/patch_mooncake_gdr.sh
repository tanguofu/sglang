#!/bin/bash
# Patch Mooncake source for GDR support on AMD MI308X + bnxt_re
# Three patches:
# 1. rdma_context.cpp fallback branch: add HipDeviceGuard before ibv_reg_mr on GPU memory
# 2. worker_pool.cpp transferWorker: add hipSetDevice in worker thread entry
# 3. CMakeLists.txt: propagate USE_HIP_DMABUF to rdma_transport OBJECT target
set -euo pipefail

MOONCAKE_SRC="/sgl-workspace/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport"

echo "=== Patch 1: rdma_context.cpp fallback branch (line 458-464) ==="
RDMA_CTX="${MOONCAKE_SRC}/rdma_context.cpp"
# Backup
cp "${RDMA_CTX}" "${RDMA_CTX}.bak.gdr"

# Patch the fallback branch: add HipDeviceGuard + hipSetDevice before ibv_reg_mr
# Original (lines 458-464):
#   } else if (hipAttr.type == hipMemoryTypeDevice &&
#              !isKernelDmabufSupported()) {
#       // Kernel lacks CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY —
#       // ibv_reg_dmabuf_mr may succeed but transfers will silently fail.
#       // Fail at registration time instead.
#       mrMeta.addr = addr;
#       mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);
python3 -c "
import re
with open('${RDMA_CTX}', 'r') as f:
    content = f.read()

old = '''    } else if (hipAttr.type == hipMemoryTypeDevice &&
               !isKernelDmabufSupported()) {
        // Kernel lacks CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY —
        // ibv_reg_dmabuf_mr may succeed but transfers will silently fail.
        // Fail at registration time instead.
        mrMeta.addr = addr;
        mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);'''

new = '''    } else if (hipAttr.type == hipMemoryTypeDevice &&
               !isKernelDmabufSupported()) {
        // Kernel lacks CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY —
        // ibv_reg_dmabuf_mr may succeed but transfers will silently fail.
        // Fail at registration time instead.
        //
        // GDR PATCH: Set HIP device context before ibv_reg_mr on GPU memory.
        // Without this, worker threads (which don't inherit device context
        // from the main thread) will cause ibv_reg_mr to hang or fault.
        HipDeviceGuard dev_guard(hipAttr.device);
        if (!dev_guard.set_ok()) {
            LOG(ERROR) << \"Failed to set HIP device to \" << hipAttr.device
                       << \" for ibv_reg_mr fallback of \" << (uintptr_t)addr;
            return ERR_CONTEXT;
        }
        mrMeta.addr = addr;
        mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);'''

if old in content:
    content = content.replace(old, new)
    with open('${RDMA_CTX}', 'w') as f:
        f.write(content)
    print('Patch 1 applied: rdma_context.cpp fallback branch')
else:
    print('WARNING: Patch 1 pattern not found — may already be patched')
    # Check if already patched
    if 'GDR PATCH: Set HIP device context before ibv_reg_mr' in content:
        print('  -> Already patched (GDR PATCH comment found)')
    else:
        print('  -> ERROR: Pattern not found and not already patched!')
        exit(1)
"

echo "=== Patch 2: worker_pool.cpp transferWorker (add hipSetDevice) ==="
WORKER_POOL="${MOONCAKE_SRC}/worker_pool.cpp"
cp "${WORKER_POOL}" "${WORKER_POOL}.bak.gdr"

python3 -c "
with open('${WORKER_POOL}', 'r') as f:
    content = f.read()

old = '''void WorkerPool::transferWorker(int thread_id) {
    bindToSocket(numa_socket_id_);
    const static uint64_t kWaitPeriodInNano = 100000000;  // 100ms'''

new = '''void WorkerPool::transferWorker(int thread_id) {
    bindToSocket(numa_socket_id_);
    // GDR PATCH: Set HIP device context in worker thread.
    // Worker threads don't inherit device context from the main thread.
    // Without hipSetDevice, ibv_reg_mr on GPU memory (GDR fallback path)
    // will hang or cause \"Memory access fault by GPU node-X\".
    (void)hipSetDevice(0);
    const static uint64_t kWaitPeriodInNano = 100000000;  // 100ms'''

if old in content:
    content = content.replace(old, new)
    with open('${WORKER_POOL}', 'w') as f:
        f.write(content)
    print('Patch 2 applied: worker_pool.cpp transferWorker hipSetDevice')
else:
    print('WARNING: Patch 2 pattern not found — may already be patched')
    if 'GDR PATCH: Set HIP device context in worker thread' in content:
        print('  -> Already patched')
    else:
        print('  -> ERROR: Pattern not found and not already patched!')
        exit(1)
"

echo "=== Patch 3: CMakeLists.txt USE_HIP_DMABUF propagation ==="
CMAKE_FILE="${MOONCAKE_SRC}/CMakeLists.txt"
cp "${CMAKE_FILE}" "${CMAKE_FILE}.bak.gdr"

python3 -c "
with open('${CMAKE_FILE}', 'r') as f:
    content = f.read()

old = '''file(GLOB RDMA_SOURCES \"*.cpp\")

add_library(rdma_transport OBJECT \${RDMA_SOURCES})
target_link_libraries(rdma_transport PRIVATE JsonCpp::JsonCpp glog::glog
                                             pthread)'''

new = '''file(GLOB RDMA_SOURCES \"*.cpp\")

add_library(rdma_transport OBJECT \${RDMA_SOURCES})
target_link_libraries(rdma_transport PRIVATE JsonCpp::JsonCpp glog::glog
                                             pthread)
# GDR PATCH: Propagate USE_HIP_DMABUF from parent target to rdma_transport.
# Without this, the OBJECT target doesn't inherit the define and the GDR
# code path (HipDeviceGuard, isKernelDmabufSupported) is compiled out.
if(USE_HIP_DMABUF)
    target_compile_definitions(rdma_transport PUBLIC USE_HIP_DMABUF)
endif()
if(USE_HIP)
    target_compile_definitions(rdma_transport PUBLIC USE_HIP)
endif()'''

if old in content:
    content = content.replace(old, new)
    with open('${CMAKE_FILE}', 'w') as f:
        f.write(content)
    print('Patch 3 applied: CMakeLists.txt USE_HIP_DMABUF propagation')
else:
    print('WARNING: Patch 3 pattern not found — may already be patched')
    if 'GDR PATCH: Propagate USE_HIP_DMABUF' in content:
        print('  -> Already patched')
    else:
        print('  -> ERROR: Pattern not found and not already patched!')
        exit(1)
"

echo "=== All patches applied. Verifying... ==="
echo "--- rdma_context.cpp GDR PATCH ---"
grep -n "GDR PATCH" "${RDMA_CTX}" | head -5
echo "--- worker_pool.cpp GDR PATCH ---"
grep -n "GDR PATCH\|hipSetDevice" "${WORKER_POOL}" | head -5
echo "--- CMakeLists.txt GDR PATCH ---"
grep -n "GDR PATCH\|USE_HIP_DMABUF" "${CMAKE_FILE}" | head -5
echo "=== Done ==="
