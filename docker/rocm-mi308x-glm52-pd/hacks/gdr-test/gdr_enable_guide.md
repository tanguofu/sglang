# MI308X (gfx942) 开启 GPU-Direct RDMA (GDR) 实操指南

> **目标**：让 SGLang 1P1D 跨节点 PD 分离的 KV 传输走真正的 GPU-Direct RDMA（NIC 直接 DMA GPU 显存），绕开 `SGLANG_PD_HOST_STAGING=1` 的 D2H/H2D bounce buffer 拷贝。
> **适用环境**：MI308X (gfx942) + TencentOS Server 3.1 (kernel 5.4.119) + bnxt_re RoCE v2 NIC + Mooncake transfer engine (USE_HIP_DMABUF=ON) + SGLang PD disagg。
> **作者**：guofutan
> **最后更新**：2026-07-21
> **相关文档**：
> - [MI308X 跨机 PD 分离 RDMA 排查记录](/p/4025463879)（2026-07-12 排查过程）
> - [MI308X GDR 验证与 Mooncake 代码根因分析](/p/4026859496)（2026-07-21 根因 + 验证）

---

## 1. 前置结论：本节点 GDR 可用性

### 1.1 已验证事实（2026-07-21）

| 检查项 | 状态 | 证据 |
|--------|------|------|
| amdgpu peermem 加载 | ✅ | `/proc/kallsyms` 有 `kfd_init_peer_direct [amdgpu]` |
| `ib_register_peer_memory_client` 导出 | ✅ | `/proc/kallsyms` 有 `T ib_register_peer_memory_client [ib_core]` |
| bnxt_re peer client | ✅ | `/proc/kallsyms` 有 `bnxt_re_get_peer_mem [bnxt_re]` |
| `ibv_reg_mr` 在 GPU 显存上 | ✅ | Rust 测试: lkey/rkey 有效, MR addr == GPU ptr |
| loopback RDMA WRITE from GPU | ✅ | Rust 测试: 4MB GPU→host, status=0, 数据校验通过 |
| `CONFIG_PCI_P2PDMA` | ❌ 未开启 | `/proc/kallsyms` 无 `pci_p2pdma` 符号 |
| `CONFIG_DMABUF_MOVE_NOTIFY` | ❌ 未开启 | `/proc/kallsyms` 无 `dma_buf_move_notify` 符号 |

**关键结论**：amdgpu peermem 走 `ib_register_peer_memory_client` 路径，**不依赖** `CONFIG_PCI_P2PDMA`。loopback GDR 已证实可用。跨节点失败的根因不是内核缺 P2PDMA，而是 **Mooncake 代码的 device context 缺失**（见第 3 节）。

### 1.2 为什么之前认为"必须升内核/重编 OFED"

2026-07-12 排查时，`nm ib_core.ko | grep peer` 查不到 `ib_register_peer_memory_client` 导出，误判为 OFED 缺 peer_memory。实际原因可能是：
1. 当时查的是 `.ko` 文件（未加载状态的符号表可能与运行时不同）
2. 节点不同（07-12 是 152/172，验证节点是 144）
3. 驱动/OFED 版本更新

**当前环境已具备 GDR 所需的内核侧能力**，无需升内核、无需重编 OFED。

---

## 2. 开启 GDR 的完整步骤

### 步骤 1：确认环境前置条件

在 prefill 和 decode pod 上分别执行：

```bash
# 1. peermem 符号齐全（三条都要有输出）
grep -E "ib_register_peer_memory_client|kfd_init_peer_direct|bnxt_re_get_peer_mem" /proc/kallsyms
# 预期:
#   xxx T ib_register_peer_memory_client  [ib_core]
#   xxx t kfd_init_peer_direct            [amdgpu]
#   xxx t bnxt_re_get_peer_mem            [bnxt_re]

# 2. bnxt_re 网卡 PORT_ACTIVE
ibv_devinfo -d bnxt_re_bond0 | grep state
# 预期: state: PORT_ACTIVE (4)

# 3. Mooncake 已编译 USE_HIP_DMABUF=ON
grep USE_HIP_DMABUF /sgl-workspace/Mooncake/build/CMakeCache.txt
# 预期: USE_HIP_DMABUF:BOOL=ON
```

如果第 1 步符号缺失，说明该节点 OFED 确实缺 peer_memory 支持，需走第 6 节的内核侧修复路径。

### 步骤 2：应用 Mooncake 代码补丁（核心修复）

**这是开启跨节点 GDR 的关键**。Mooncake 的 fallback 分支（内核无 P2PDMA 时走的路径）缺 `HipDeviceGuard`，导致跨节点传输时 peermem driver 用错误的 GPU BAR，触发 "Memory access fault by GPU node-X"。

需要改 2 个文件，共 ~15 行代码。

#### 补丁 2.1：`rdma_context.cpp` fallback 分支加 HipDeviceGuard

**文件**：`mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp`
**位置**：第 459-463 行（`USE_HIP_DMABUF` 分支内，`hipMemoryTypeDevice && !isKernelDmabufSupported()`）

**改前**（第 458-463 行）：
```cpp
    } else if (hipAttr.type == hipMemoryTypeDevice &&
               !isKernelDmabufSupported()) {
        // Kernel lacks CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY —
        // ibv_reg_dmabuf_mr may succeed but transfers will silently fail.
        // Fail at registration time instead.
        mrMeta.addr = addr;
        mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);
```

**改后**：
```cpp
    } else if (hipAttr.type == hipMemoryTypeDevice &&
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
        mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);
```

**原理**：相邻的 dmabuf 分支（第 467 行）和 `#else` staging 分支（第 521 行）都有 `HipDeviceGuard(hipAttr.device)`，唯独这个 fallback 分支漏了。补上后，`ibv_reg_mr` 在 GPU addr 上注册时，当前线程的 HIP device 会切到 `addr` 所属的 GPU，peermem driver 就能用正确的 BAR 做 DMA 映射。

#### 补丁 2.2：`worker_pool.cpp` worker 线程加 hipSetDevice

**文件**：`mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp`
**位置**：第 512-513 行（`transferWorker` 函数入口）

**改前**（第 512-513 行）：
```cpp
void WorkerPool::transferWorker(int thread_id) {
    bindToSocket(numa_socket_id_);
```

**改后**：
```cpp
void WorkerPool::transferWorker(int thread_id) {
    bindToSocket(numa_socket_id_);
    // CRITICAL: worker threads do not inherit the caller's HIP device
    // context. Without hipSetDevice, the default device (0) is used,
    // which causes "Memory access fault by GPU node-X" when posting
    // sends for slices on other GPUs. Set device 0 as a safe baseline;
    // registerMR's HipDeviceGuard pins the correct device per-buffer,
    // and ibv_post_send reuses the MR's lkey which already has the
    // correct BAR mapping.
    (void)hipSetDevice(0);
```

**原理**：`transferWorker` 只调 `bindToSocket`（NUMA 绑定），不调 `hipSetDevice`。worker 线程的 HIP device context 未定义（默认 0）。补丁 2.1 已经在 `ibv_reg_mr` 时通过 `HipDeviceGuard` 切到正确 device，MR 的 lkey 已经绑定了正确的 BAR。`ibv_post_send` 复用这个 lkey，所以 worker 线程只需有一个合法的 device context（不一定是 slice 所属的 device）即可——因为 DMA 映射在 MR 注册时已经完成。

> **注**：如果补丁 2.1 + 2.2 后仍有跨节点 fault，可能需要更彻底的修复：在 `submitPostSend`（第 94 行）按 slice 的 source_addr 反查 device 并 `hipSetDevice`。但这需要调用 `hipPointerGetAttributes` 有额外开销，建议先试 2.1+2.2。

### 步骤 3：重新编译 Mooncake

在 pod 内或构建机上：

```bash
cd /sgl-workspace/Mooncake/build
cmake .. -DUSE_HIP_DMABUF=ON
make -j$(nproc)
# 产物: mooncake-transfer-engine/libmooncake-transfer-engine.so
```

确认补丁生效：
```bash
# 检查 fallback 分支有 HipDeviceGuard
grep -A 3 "hipMemoryTypeDevice" /sgl-workspace/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp | grep HipDeviceGuard
# 预期: HipDeviceGuard dev_guard(hipAttr.device);

# 检查 transferWorker 有 hipSetDevice
grep -A 2 "transferWorker" /sgl-workspace/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp | grep hipSetDevice
# 预期: (void)hipSetDevice(0);
```

### 步骤 4：关闭 SGLANG_PD_HOST_STAGING

**文件**：`sglang-1p1d-charts/templates/_helpers.tpl`
**位置**：第 14-22 行

**改前**：
```yaml
# Host staging: RDMA via pinned host memory (GPU<->CPU hipMemcpy bounce).
# Required because kernel lacks CONFIG_PCI_P2PDMA ...
- name: SGLANG_PD_HOST_STAGING
  value: "1"
```

**改后**（注释保留历史，值改为 "0"）：
```yaml
# GDR enabled: Mooncake ibv_reg_mr on GPU memory + HipDeviceGuard patch
# allows NIC to DMA directly from GPU VRAM (no D2H/H2D bounce).
# Set to "1" only as fallback if cross-node GDR faults occur.
- name: SGLANG_PD_HOST_STAGING
  value: "0"
```

### 步骤 5：重新部署并验证

```bash
# 1. 重新部署 prefill + decode（确保新 Mooncake .so 生效）
helm upgrade sglang-1p1d ./sglang-1p1d-charts -n kube-system

# 2. 等 pod ready
kubectl get pods -n kube-system | grep sglang-1p1d

# 3. 检查日志不再有 "Memory access fault"
kubectl logs -n kube-system sglang-1p1d-decode-0 | grep -i "memory access fault"
# 预期: 无输出

# 4. 检查日志不再有 host staging（说明走 GDR 了）
kubectl logs -n kube-system sglang-1p1d-decode-0 | grep -i "host staging"
# 预期: 无 "registered N host buffers" 输出

# 5. rdma_context.cpp:136 的 WARNING 仍会出现（内核确实没 P2PDMA），但
#    不影响功能——走的是 fallback 分支带 HipDeviceGuard，peermem 仍可用
kubectl logs -n kube-system sglang-1p1d-decode-0 | grep "CONFIG_PCI_P2PDMA"
# 预期: WARNING ... falling back to ibv_reg_mr() ...（正常）

# 6. 跑 PD 请求验证
# 短请求（18 token）
curl -X POST http://<router-ip>:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"你好"}],"max_tokens":18}'

# 长请求（256+ token）
curl -X POST http://<router-ip>:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"请详细介绍智谱AI的发展历史"}],"max_tokens":256}'
```

### 验证成功的标志

| 标志 | 说明 |
|------|------|
| 无 "Memory access fault by GPU node-X" | peermem 用对了 BAR |
| 无 "Host staging: registered N host buffers" | 走 GDR 而非 host staging |
| 无 "KVTransferError: Aborted by AbortReq" | KV 传输未超时 |
| 短请求 200 + 答案正确 | 基本功能 OK |
| 长请求 200 + 答案正确 | 大块 KV 传输 OK |
| `rdma_context.cpp:136` WARNING 仍出现 | 正常（内核确实没 P2PDMA，走 fallback 带 guard） |

---

## 3. 根因解析：为什么需要这两个补丁

### 3.1 Mooncake GDR 注册的三条分支

`rdma_context.cpp:440-475` 根据 `hipPointerGetAttributes` 的结果分三条路径：

| 条件 | 行号 | HipDeviceGuard? | 走的路径 |
|------|------|-----------------|----------|
| Host memory | 444-447 | ❌ 不需要 | `ibv_reg_mr` 标准 host 路径 |
| Managed memory | 453-457 | ❌ | `ibv_reg_mr`（fallback，pages 可能迁移） |
| **Device memory + `!isKernelDmabufSupported()`** | **459-463** | **❌ 缺失！** | **`ibv_reg_mr` on GPU addr（fallback 分支）** |
| Device memory + `isKernelDmabufSupported()` | 465-475 | ✅ `HipDeviceGuard(hipAttr.device)` | `hsa_amd_portable_export_dmabuf` + `ibv_reg_dmabuf_mr`（dmabuf GDR） |

**当前内核无 P2PDMA → `isKernelDmabufSupported()` 返回 false → 走第 459-463 行 fallback 分支**。这个分支是唯一处理 GPU device memory 但不用 `HipDeviceGuard` 的分支。

### 3.2 跨节点失败的根因链

```
Mooncake worker 线程 (worker_pool.cpp:512)
  └─ bindToSocket(numa_socket_id_)     ← 只绑 NUMA，不设 GPU device
  └─ 无 hipSetDevice                   ← 线程 device context = 默认 0

ibv_reg_mr on GPU addr (rdma_context.cpp:459-463, fallback 分支)
  └─ 无 HipDeviceGuard                 ← 不切到 addr 所属的 GPU device
  └─ peermem driver 用当前 device(0) 的 BAR 映射 GPU 内存
  └─ 若 addr 属于 GPU 5，但用 device 0 的 BAR → 错误映射

ibv_post_send (跨节点 RDMA WRITE)
  └─ NIC DMA 从错误 BAR 读 GPU 内存
  └─ "Memory access fault by GPU node-5"
```

### 3.3 为什么 loopback 测试成功

Rust GDR 测试程序（[验证文档](/p/4026859496)）：
- 单线程：`hipSetDevice(0)` 在 main 里调用一次，整个线程在 device 0
- 单 GPU：只用 GPU 0，NIC DMA 从 GPU 0 的 VRAM，device context 正确
- 所以 peermem 用对 BAR → DMA 成功

Mooncake 生产场景：
- 多线程：worker 线程不继承主线程的 device context
- 8 GPU：`HIP_VISIBLE_DEVICES=0..7`，worker 默认 device 0
- rank 5 的 worker 传输 GPU 5 的 KV cache 时，device context 错位 → fault

### 3.4 补丁如何修复

- **补丁 2.1**：`ibv_reg_mr` 注册时用 `HipDeviceGuard(hipAttr.device)` 切到 addr 所属 GPU → peermem 用正确 BAR → MR 的 lkey 绑定正确 BAR
- **补丁 2.2**：worker 线程设一个合法 device context（0 即可）→ `ibv_post_send` 复用 MR 的 lkey（BAR 已在注册时固定）→ DMA 正确

---

## 4. 环境变量配置汇总

开启 GDR 后，`_helpers.tpl` 的关键 env：

```yaml
- name: MOONCAKE_PROTOCOL
  value: "rdma"                    # 必须 rdma
- name: MC_GID_INDEX
  value: "3"                       # RoCE v2 GID index（按实际 ibv_query_gid 结果选）
- name: MC_DISABLE_HIP_TRANSPORT
  value: "1"                       # 禁用 HIP IPC（跨节点不用）
- name: SGLANG_PD_HOST_STAGING
  value: "0"                       # ← 关闭 host staging，走 GDR
- name: HIP_VISIBLE_DEVICES
  value: "0,1,2,3,4,5,6,7"        # 8 GPU
```

保留不变的环境变量（与 GDR 无关，但需保持）：
```yaml
- name: NCCL_DEBUG
  value: "WARN"
- name: HSA_ENABLE_SDMA
  value: "0"
- name: HIP_FORCE_DEV_KERNARG
  value: "1"
- name: PYTORCH_CUDA_ALLOC_CONF
  value: "expandable_segments:True"
- name: PYTORCH_ROCM_ARCH
  value: "gfx942"
```

---

## 5. 回退方案

如果补丁后跨节点 GDR 仍 fault（极少数情况，可能是其他 device context 问题）：

```yaml
# _helpers.tpl 临时回退
- name: SGLANG_PD_HOST_STAGING
  value: "1"                       # 回到 host staging
```

无需重新编译 Mooncake——`SGLANG_PD_HOST_STAGING=1` 会让 SGLang 在 `conn.py` 层走 D2H/H2D bounce buffer 路径，绕过 Mooncake 的 GPU MR 注册。

---

## 6. 内核侧修复路径（非必需，仅当 peermem 符号缺失）

> **注意**：当前环境 peermem 符号齐全（步骤 1 已验证），**不需要**做本节操作。本节仅供 peermem 缺失的节点参考。

### 6.1 检查 peermem 是否缺失

```bash
grep "ib_register_peer_memory_client" /proc/kallsyms
# 无输出 = ib_core 没导出该符号 = OFED 编译时未开 peer_memory
```

### 6.2 方案 A：重编 OFED ib_core 带 peer_mem

```bash
# 进宿主（需 hostPID + privileged 调试 pod）
chroot /host bash
cd /usr/src/ofa_kernel-5.8/source
./configure --help | grep -i peer      # 查 peer_mem 编译开关
# 重编 ib_core.ko 带 CONFIG_INFINIBAND_PEER_MEM
make
# replace + reload
rmmod bnxt_re; rmmod ib_core
cp drivers/infiniband/core/ib_core.ko /lib/modules/$(uname -r)/extra/mlnx-ofa_kernel/drivers/infiniband/core/
modprobe ib_core; modprobe bnxt_re
# 验证
nm ib_core.ko | grep ib_register_peer_memory_client   # 应有 T
```

### 6.3 方案 B：装 Broadcom netxtreme-peer-mem

```bash
yum list available | grep -i peer
yum install netxtreme-peer-mem
modprobe ib_peer_mem
```

### 6.4 方案 C：AMD Direct GMA（绕过 P2PDMA）

```bash
# 内核启动参数加 amdgpu.direct_gma_size=96，需重启节点
# 前提：IOMMU pass-through（iommu=pt amd_iommu=on，当前已满足）
# Direct GMA 用 dma_map_resource，不依赖 CONFIG_PCI_P2PDMA
grubby --update-kernel=ALL --args="amdgpu.direct_gma_size=96"
reboot
```

**优先级**：当前环境无需本节。若未来遇到 peermem 缺失节点，优先方案 B（yum 装），其次方案 A（重编），最后方案 C（重启节点）。

---

## 7. isKernelDmabufSupported() 的已知 bug

`rdma_context.cpp:103-106` 跳过所有 `.gz` 文件：

```cpp
if (path.find(".gz") != std::string::npos) continue;
```

但 `/proc/config.gz` 是很多 pod **唯一的** kernel config 来源（`/boot/config-*` 不存在）。这导致即使内核有 P2PDMA（config.gz 里写了 `CONFIG_PCI_P2PDMA=y`），也因跳过 gz 而误判为 false。

**当前影响**：本节点内核确实没 P2PDMA，所以结论碰巧正确（走 fallback 分支是对的）。

**修复（可选）**：用 `popen("zcat /proc/config.gz")` 解压读取：

```cpp
if (path.find(".gz") != std::string::npos) {
    FILE* pipe = popen(("zcat " + path).c_str(), "r");
    if (pipe) {
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe)) {
            std::string line(buffer);
            for (size_t i = 0; i < 2; ++i) {
                if (!found[i] && line.find(needles[i]) != std::string::npos)
                    found[i] = true;
            }
            if (found[0] && found[1]) break;
        }
        pclose(pipe);
    }
    continue;
}
```

**优先级低**：不影响当前 GDR 功能（fallback 分支带 HipDeviceGuard 后已可用）。

---

## 8. 验证 GDR 可用性的独立测试程序

如需在不跑 SGLang 的情况下验证节点 GDR 是否可用，可用 Rust + C shim 测试程序：

**文件**：
- `/tmp/gdr-test/main.rs` — Rust 驱动（GPU 分配 + ibv_reg_mr + 调 shim）
- `/tmp/gdr-test/build.rs` — 链接 libibverbs + libamdhip64 + libshim
- `/tmp/shim.c` → `/tmp/libshim.so` — C shim（处理 verbs 复杂结构体）

**运行**：
```bash
kubectl exec -n kube-system sglang-1p1d-decode-0 -- \
  bash -c 'LD_LIBRARY_PATH=/tmp /tmp/gdr-test/target/release/gdr-test'
```

**预期输出**：
```
=== GDR (GPU-Direct RDMA) test (via C shim) ===
[3] Registering GPU memory with ibv_reg_mr...
    GPU MR: lkey=xxx rkey=xxx addr=0x... (matches GPU ptr: true)
[7] RDMA WRITE from GPU buffer to host buffer (THE GDR TEST)...
    ibv_post_send OK
[8] Polling CQ for completion...
    poll_cq: n=1 status=0 byte_len=0
    RDMA WRITE completed successfully!
[9] Verifying data in host buffer...
    Data verified: host buffer matches GPU pattern 0xA5
=== GDR TEST PASSED: GPU-Direct RDMA WORKS! ===
```

详见 [GDR 验证文档](/p/4026859496)第 2 节。

---

## 9. 常见问题

### Q1: 补丁后仍有 "Memory access fault"

**排查**：
1. 确认 Mooncake .so 已更新（`ldd` 检查链接的是新库）
2. 确认 `SGLANG_PD_HOST_STAGING=0`（否则走 host staging 不经 fallback 分支）
3. 检查 fault 的 GPU node 编号：如果是 node-0，可能是 worker 线程的 `hipSetDevice(0)` 不够，需要在 `submitPostSend` 按 slice 的 source_addr 反查 device 并设 device
4. 临时回退 `SGLANG_PD_HOST_STAGING=1` 保持可用

### Q2: `rdma_context.cpp:136` 的 WARNING 要处理吗

```
W rdma_context.cpp:136] Kernel lacks CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY
  ... falling back to ibv_reg_mr() ...
```

**不用处理**。这是 `isKernelDmabufSupported()` 返回 false 的正常告警。fallback 分支带 HipDeviceGuard 后已可用。这个 WARNING 只是说明没走 dmabuf 路径，不代表 GDR 不可用。

### Q3: 为什么不用 dmabuf 路径（`ibv_reg_dmabuf_mr`）

dmabuf 路径需要内核 `CONFIG_PCI_P2PDMA=y` + `CONFIG_DMABUF_MOVE_NOTIFY=y`，当前内核都没开启。开启需要重编内核或升级内核版本（≥6.x 通常默认开）。

而 fallback 路径（`ibv_reg_mr` on GPU addr）走 amdgpu peermem（`ib_register_peer_memory_client`），**不需要 P2PDMA**。两者都能实现 GDR，只是内核侧机制不同：
- dmabuf 路径：标准 Linux dmabuf + P2PDMA（内核通用框架）
- peermem 路径：厂商特定（amdgpu kfd_peerdirect），通过 `ib_register_peer_memory_client` 注册 peer memory client

### Q4: host staging 和 GDR 的性能差异

- **host staging**：每次 KV 传输有 D2H（GPU→host `hipMemcpy`）+ RDMA + H2D（host→GPU `hipMemcpy`），额外 2 次拷贝 + 2 次跨 PCIe DMA
- **GDR**：NIC 直接 DMA GPU 显存，0 次拷贝

预期 GDR 比 host staging 快 30-50%（取决于 KV 块大小和 PCIe 带宽），需实测确认。

### Q5: 单节点 PD 能用 GDR 吗

能。loopback GDR 已验证可用。如果 prefill + decode 在同一节点（KV 传输走 loopback QP），无需补丁 2.1/2.2（因为单进程内 device context 通常正确）。但生产 1P1D 通常跨节点，需要补丁。

---

## 10. 改动清单速查

| 序号 | 文件 | 行号 | 改动 | 必需? |
|------|------|------|------|-------|
| 1 | `Mooncake/.../rdma_context.cpp` | 459-463 | fallback 分支加 `HipDeviceGuard(hipAttr.device)` | ✅ 跨节点必需 |
| 2 | `Mooncake/.../worker_pool.cpp` | 512-513 | `transferWorker` 入口加 `(void)hipSetDevice(0);` | ✅ 跨节点必需 |
| 3 | `sglang-1p1d-charts/templates/_helpers.tpl` | 21-22 | `SGLANG_PD_HOST_STAGING` 改 `"0"` | ✅ 开启 GDR 必需 |
| 4 | `Mooncake/.../rdma_context.cpp` | 103-106 | `isKernelDmabufSupported` 读 `/proc/config.gz` | ❌ 可选（修 bug，当前不影响） |

补丁 1+2+3 完成后，重新编译 Mooncake + 重新部署 SGLang，即可走跨节点 GDR。

---

*本文档为 MI308X 开启 GDR 的实操指南。核心结论：当前环境 peermem 已就绪，无需升内核/重编 OFED，只需补 Mooncake 的 HipDeviceGuard + hipSetDevice 两处代码即可开启跨节点 GDR。*
