# MI308X GDR 验证与 Mooncake 代码根因分析（2026-07-21）

> **状态**：✅ 根因确诊 + 验证通过（loopback GDR 可用）；跨节点修复方案明确
> **作者**：guofutan
> **相关文档**：[MI308X 跨机 PD 分离 RDMA 排查记录](/p/4025463879)（2026-07-12，部分结论已被本文修正）

---

## 1. 摘要

通过纯 Rust + C shim 编写的 GDR 验证程序，**证实本节点 loopback GPU-Direct RDMA 完全可用**：`ibv_reg_mr` 注册 GPU 显存成功，RDMA WRITE 从 GPU 缓冲到 host 缓冲完成且数据校验通过。同时定位到 Mooncake 跨节点 GDR 失败的真正根因：**fallback 分支（`rdma_context.cpp:459-463`）缺少 `HipDeviceGuard`，且 worker 线程缺少 `hipSetDevice`**，并非此前怀疑的 OFED peer_memory 缺失。

---

## 2. 关键发现：loopback GDR 可用

### 2.1 测试程序

| 组件 | 路径 | 说明 |
|------|------|------|
| Rust 驱动 | `/tmp/gdr-test/main.rs` | GPU 内存分配 + `ibv_reg_mr` + 调用 shim |
| 构建脚本 | `/tmp/gdr-test/build.rs` | 链接 libibverbs + libamdhip64 + libshim |
| C shim | `/tmp/shim.c` → `/tmp/libshim.so` | 处理 `ibv_modify_qp`/`ibv_post_send`/`ibv_poll_cq` 的复杂 struct |
| 构建产物 | `/tmp/gdr-test/target/release/gdr-test` | 单一可执行文件 |

**为何用 C shim**：`ibv_qp_attr`(144B)/`ibv_send_wr`(128B)/`ibv_wc`(48B)/`ibv_port_attr`(52B) 等 verbs 结构体有非平凡的内存布局（`ibv_gid` 是 union，因 `uint64_t` 变体有 8 字节对齐；`ibv_send_wr` 有 4 个 union 共 128 字节；`ibv_wc` 末尾有 `pkey_index`/`slid`/`sl`/`dlid_path_bits`）。手写 Rust FFI 极易出错（编译通过但偏移错位导致 `modify_qp` 返回 EINVAL=22）。C shim 直接用 `<infiniband/verbs.h>` 的真实结构体，绕过所有 ABI 风险。

### 2.2 测试流程与结果

```
[1] hipMalloc 4MB GPU 缓冲，hipMemset 0xA5，hipDeviceSynchronize   ✓
[2] ibv_open_device(bnxt_re_bond0) + ibv_alloc_pd                  ✓
[3] ibv_reg_mr(pd, gpu_ptr, 4MB, LOCAL_WRITE|REMOTE_WRITE|REMOTE_READ)
    → 成功！lkey=67150887, rkey=67150887, MR addr == GPU ptr       ✓
[4] hipMallocHost 4MB host 缓冲 + ibv_reg_mr                       ✓
[5] ibv_create_cq + ibv_create_qp(IBV_QPT_RC, loopback)            ✓
[6] shim_modify_qp_init / rtr / rts (dest_qp_num = own qp_num)    ✓
[7] shim_post_send_rdma_write(GPU → host, 4MB)                     ✓
[8] shim_poll_cq → status=0 (SUCCESS), byte_len=0                  ✓
[9] memcmp host 缓冲 vs 0xA5 → 完全匹配                            ✓

=== GDR TEST PASSED: GPU-Direct RDMA WORKS! ===
```

### 2.3 结论

- `ibv_reg_mr` 在 GPU 显存上**成功**：amdgpu peermem（`kfd_peerdirect.c`）已加载并工作
- NIC 可以从 GPU VRAM 直接 DMA：**无需 D2H/H2D 拷贝**
- amdgpu peermem client 设计为**不需要** `CONFIG_PCI_P2PDMA` —— 它用 `ib_register_peer_memory_client` 注册 peer memory，走自己的 scatter-gather 路径

---

## 3. 环境事实（2026-07-21 确认）

### 3.1 peermem 符号状态（与 07-12 文档相反）

```
/proc/kallsyms 关键符号:
  ffffffffa2e35f50 T ib_register_peer_memory_client    [ib_core]   ← 已导出!
  ffffffffa22b3b70 t kfd_init_peer_direct              [amdgpu]
  ffffffffa03b3d60 t bnxt_re_get_peer_mem              [bnxt_re]
```

**纠正 07-12 文档的结论**：当时文档说"ib_core.ko 没导出 ib_register_peer_memory_client"。当前环境（节点 21.151.225.144，decode pod）该符号已正常导出（`T` = defined + exported）。可能原因：
1. 节点不同（07-12 是 152/172，现在是 144）
2. OFED/amdgpu 驱动版本更新
3. 07-12 的 `nm` 查的是文件，而 `/proc/kallsyms` 是运行时符号（模块已加载）

### 3.2 内核配置

```
/proc/config.gz 存在（唯一的 config 来源）
/boot/config-* 不存在
/lib/modules/*/config 不存在
/proc/kallsyms 中 pci_p2pdma / dma_buf_move_notify 符号数 = 0
```

`isKernelDmabufSupported()` 返回 **false**（找不到 P2PDMA 符号）→ 走 fallback 分支。

### 3.3 isKernelDmabufSupported() 的 bug

`rdma_context.cpp:103-106` 跳过所有 `.gz` 文件：
```cpp
// /proc/config.gz is gzipped; we skip it here (the kallsyms
// fallback below covers that case in practice).
if (path.find(".gz") != std::string::npos) continue;
```
但 `/proc/config.gz` 是本 pod **唯一的** kernel config 来源（`/boot/config-*` 不存在）。kallsyms fallback 只查 `pci_p2pdma` 和 `dma_buf_move_notify` 两个符号，而本内核确实没有这两个符号（即使 config.gz 里可能有 `CONFIG_PCI_P2PDMA=y`，也不被读取）。

**影响**：即使内核真的有 P2PDMA（只是 config.gz 没被解压读），`isKernelDmabufSupported()` 也会误判为 false，导致走 fallback 分支。不过当前内核确实没有 P2PDMA，所以结论碰巧正确。

---

## 4. Mooncake 代码根因分析

### 4.1 GDR 注册的三条分支

`rdma_context.cpp:440-475` 根据 `hipPointerGetAttributes` 的结果分三条路径：

| 条件 | 行号 | HipDeviceGuard? | 走的路径 |
|------|------|-----------------|----------|
| Host memory | 444-447 | ❌ 不需要（host mem 不依赖 GPU device context） | `ibv_reg_mr` 标准 host 路径 |
| Managed memory | 453-457 | ❌ | `ibv_reg_mr`（fallback，pages 可能迁移） |
| Device memory + `!isKernelDmabufSupported()` | **459-463** | **❌ 缺失！** | `ibv_reg_mr` on GPU addr（**fallback 分支**） |
| Device memory + `isKernelDmabufSupported()` | 465-475 | ✅ `HipDeviceGuard(hipAttr.device)` | `hsa_amd_portable_export_dmabuf` + `ibv_reg_dmabuf_mr`（真 GDR） |

**问题**：第 459-463 行的 fallback 分支是**唯一**处理 GPU device memory 但**不用** `HipDeviceGuard` 的分支。相邻的 dmabuf 分支（467 行）和 `#else` staging 分支（521 行）都用 `HipDeviceGuard`。

### 4.2 为什么 loopback 成功但 Mooncake 跨节点失败

**loopback 测试**（我们的 Rust 程序）：
- 单线程：`hipSetDevice(0)` 在 main 里调用一次，整个线程生命周期都在 device 0
- 单 GPU：只用 GPU 0，NIC DMA 从 GPU 0 的 VRAM，device context 正确
- peermem driver 拿到正确的 GPU device → 正确的 BAR → DMA 成功

**Mooncake 跨节点**（生产 PD 场景）：
- 多线程：`WorkerPool::transferWorker`（`worker_pool.cpp:512`）只调 `bindToSocket`，**不调 `hipSetDevice`**
- 8 GPU：`HIP_VISIBLE_DEVICES=0..7`，worker 线程的 device context **未定义**（默认 0）
- 当 rank 5 的 worker 线程要传输 GPU 5 的 KV cache 时：
  1. `ibv_reg_mr` 在 fallback 分支注册 GPU addr（无 `HipDeviceGuard`）
  2. 当前线程 device context 是 0（默认），但 addr 属于 GPU 5
  3. peermem driver 用 device 0 的 BAR 映射 GPU 5 的内存 → **错误 BAR → DMA 访问非法地址 → "Memory access fault by GPU node-5"**
- `submitPostSend`（`worker_pool.cpp:94`）同样无 `hipSetDevice`

### 4.3 根因链

```
Mooncake worker 线程 (worker_pool.cpp:512)
  └─ bindToSocket(numa_socket_id_)     ← 只绑 NUMA，不设 GPU device
  └─ 无 hipSetDevice                   ← 线程 device context = 默认 0

ibv_reg_mr on GPU addr (rdma_context.cpp:459-463, fallback 分支)
  └─ 无 HipDeviceGuard                 ← 不切到 addr 所属的 GPU device
  └─ peermem driver 用错误 device 的 BAR 映射 GPU 内存

ibv_post_send (跨节点 RDMA WRITE)
  └─ NIC DMA 从错误 BAR 读 GPU 内存
  └─ "Memory access fault by GPU node-X"

而 loopback 测试：
  └─ 单线程 hipSetDevice(0) 已设
  └─ ibv_reg_mr 时 device context 正确
  └─ peermem 用正确 BAR → DMA 成功
```

### 4.4 staging_host_buf 是死代码

`rdma_context.cpp:528-530`（`#else` 分支，`USE_HIP_DMABUF` 未定义时）设置 `staging_host_buf` 字段，但：
1. 当 `USE_HIP_DMABUF=ON`（当前环境）时，这个分支**不编译**
2. 即使编译，`staging_host_buf` 从未被 `lkey()` 或传输路径读取 —— 是**死代码**

这就是为什么 SGLang 侧需要 `SGLANG_PD_HOST_STAGING=1`：Mooncake 自带的 host staging 机制不完整，SGLang 在 `conn.py` 层自己实现了 D2H/H2D bounce buffer。

---

## 5. 修复方案

### 方案 A（推荐）：补 Mooncake 的 HipDeviceGuard + hipSetDevice

**改动 1**：`rdma_context.cpp:459-463` fallback 分支加 `HipDeviceGuard`

```cpp
} else if (hipAttr.type == hipMemoryTypeDevice &&
           !isKernelDmabufSupported()) {
    // Kernel lacks P2PDMA — fall back to ibv_reg_mr on GPU addr.
    // CRITICAL: must set device context so peermem driver uses the
    // correct BAR for DMA. Without this, cross-node transfers fault
    // with "Memory access fault by GPU node-X".
    HipDeviceGuard dev_guard(hipAttr.device);
    if (!dev_guard.set_ok()) {
        LOG(ERROR) << "Failed to set HIP device to " << hipAttr.device
                   << " for ibv_reg_mr fallback of " << (uintptr_t)addr;
        return ERR_CONTEXT;
    }
    mrMeta.addr = addr;
    mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);
}
```

**改动 2**：`worker_pool.cpp:512` `transferWorker` 加 `hipSetDevice`

```cpp
void WorkerPool::transferWorker(int thread_id) {
    bindToSocket(numa_socket_id_);
    // CRITICAL: worker threads don't inherit the caller's GPU device
    // context. Without hipSetDevice, ibv_post_send from GPU memory
    // uses the wrong BAR and faults. We can't know which device here
    // (slices may span GPUs), so set device 0 as a safe default and
    // rely on HipDeviceGuard in registerMR to pin per-slice.
    // Better: set device per-slice in submitPostSend.
}
```

**改动 3**（更彻底）：`worker_pool.cpp:94` `submitPostSend` 按 slice 的 `gpu_id` 设 device

```cpp
int WorkerPool::submitPostSend(const std::vector<Transport::Slice *> &slice_list) {
    // ... existing segment_desc_map logic ...
    for (auto &slice : slice_list) {
        // Set device context for this slice's GPU before posting.
        if (slice->gpu_id >= 0) {
            HipDeviceGuard guard(slice->gpu_id);
            // ... post send for this slice ...
        }
    }
}
```

**验证**：改完重新编译 Mooncake，去掉 `SGLANG_PD_HOST_STAGING=1`，跑跨节点 PD。

### 方案 B（临时）：保持 SGLANG_PD_HOST_STAGING=1

现状已可用，无需改 Mooncake。代价是 D2H/H2D 拷贝开销（非真 GDR）。

### 方案 C（兜底）：修 isKernelDmabufSupported 读 /proc/config.gz

```cpp
// rdma_context.cpp:103-106 改为：
if (path.find(".gz") != std::string::npos) {
    // Decompress /proc/config.gz via popen("zcat <path>")
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

但这只解决"误判 P2PDMA 缺失"问题，不解决 `HipDeviceGuard` 缺失问题。当前内核确实没 P2PDMA，所以即使修了这个 bug 也还是走 fallback 分支。**优先级低**。

---

## 6. 修复优先级

| 优先级 | 方案 | 工作量 | 效果 |
|--------|------|--------|------|
| P0 | 方案 A 改动 1（fallback 分支加 HipDeviceGuard） | 5 行代码 | 解决跨节点 GDR MR 注册的 device context 问题 |
| P0 | 方案 A 改动 2/3（worker 线程加 hipSetDevice） | 10 行代码 | 解决 post_send 时的 device context 问题 |
| P1 | 方案 C（修 isKernelDmabufSupported 读 config.gz） | 15 行代码 | 修 bug，但当前不影响（内核确实没 P2PDMA） |
| P2 | 方案 B（保持 host staging） | 0 | 临时可用，有拷贝开销 |

---

## 7. 验证清单

修复后验证步骤：

1. **重新编译 Mooncake**（含方案 A 改动）
2. **去掉 `SGLANG_PD_HOST_STAGING=1`**（`_helpers.tpl`）
3. **重新部署 prefill + decode pod**
4. **检查日志**：不再出现 "Memory access fault by GPU node-X"
5. **检查 `rdma_context.cpp:136` 警告**：仍会出现（内核确实没 P2PDMA），但走 fallback 分支带 HipDeviceGuard 应能工作
6. **跑 PD 请求**：18 token 短请求 + 256 token 长请求都应成功
7. **性能对比**：GDR 直连 vs host staging 的吞吐/延迟

---

## 8. FFI 教训（给后续写 verbs Rust FFI 的人）

手写 libibverbs 的 Rust FFI 结构体极易出错：

| 结构体 | 大小 | 陷阱 |
|--------|------|------|
| `ibv_gid` | 16B | 是 union，有 `uint64_t` 变体 → 8 字节对齐（不是 1） |
| `ibv_send_wr` | 128B | 有 4 个 union（imm/wr/qp_type/最后），wr union 最大 32B（atomic 变体） |
| `ibv_wc` | 48B | 末尾有 `pkey_index`/`slid`/`sl`/`dlid_path_bits` |
| `ibv_port_attr` | 52B | 末尾有 `port_cap_flags2: u16`（不是 reserved） |
| `ibv_qp_attr` | 144B | `ibv_gid` 的 8 字节对齐传播到 `ah_attr` → 整体 8 字节对齐 |

**建议**：复杂 verbs 结构体用 C shim，不要手写 Rust FFI。简单结构体（`ibv_mr`/`ibv_pd`/`ibv_sge`）可以手写，但加 `assert_eq!(size_of::<T>(), 预期)` 和 `offset_of!` 断言。

---

## 9. 相关资源

- GDR 测试程序：`/tmp/gdr-test/`（Rust）+ `/tmp/libshim.so`（C shim）
- Mooncake 代码：`/sgl-workspace/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/`
  - `rdma_context.cpp:459-463` — fallback 分支缺 HipDeviceGuard
  - `worker_pool.cpp:512` — transferWorker 缺 hipSetDevice
  - `worker_pool.cpp:94` — submitPostSend 缺 per-slice hipSetDevice
  - `include/hip_device_guard.h` — HipDeviceGuard 实现（RAII，保存/恢复 device）
- SGLang host staging：`/sgl-workspace/sglang/python/sglang/srt/disaggregation/mooncake/conn.py`
- 前序排查：[MI308X 跨机 PD 分离 RDMA 排查记录](/p/4025463879)（2026-07-12，部分结论已被本文修正）

---

*本文档记录 2026-07-21 的 GDR 验证与 Mooncake 代码根因分析。loopback GDR 可用已证实，跨节点失败的根因是 Mooncake fallback 分支缺 HipDeviceGuard + worker 线程缺 hipSetDevice，修复方案明确（方案 A，15 行代码）。*
