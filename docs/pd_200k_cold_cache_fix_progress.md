# GLM-5.2 PD 200K Cold-Cache 乱码修复进度

**日期**: 2026-08-14 ~ 2026-08-17
**环境**: TKE MI308X 集群, node-132 (decode) + node-144 (prefill), sglang v0.5.17
**状态**: **已切真 GDR**（`SGLANG_PD_HOST_STAGING=0`）。Mooncake `ibv_reg_mr(GPU)+peermem` + HipDeviceGuard；廉价 L2 flush（prefill `buffer_wbl2` + 8B RDMA READ，decode `buffer_inv`）。unique-needle 64K/200K 5/5，QP 存活。无 flush 的纯 GDR 64K 会回到 `1.1.2...</think>` 乱码。

## 1. 问题现象

GLM-5.2 PD disaggregation (1p1d) 部署中,200K token cold-cache 请求产生乱码输出:
- 5 个 needle fact 全部提取失败 (0/5)
- 输出是重复的碎片化文本 ("1. NEEDLE1. NEEDLE1.2.3.3. NEEDLE1..." 这种)
- Warm cache (第二次请求相同 prompt) 则正常 (5/5)
- 短上下文 (<18K) cold cache 也正常

## 2. 根因分析

### 2.1 排除的错误假设

| 假设 | 测试结果 | 结论 |
|------|----------|------|
| DSA fused-store kernel 在长上下文 corrupt index K cache | length guard ≤4096 已部署 | 不是根因 (已修复但 200K 仍乱码) |
| `SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1` 导致 preshuffle 路径问题 | 移除后所有上下文都乱码 | 不是根因 (必须保留 =1) |
| fp8_e4m3 KV cache 精度退化 | 之前排查排除 | 不是根因 |

### 2.2 确认的根因: GPU L2 Cache 不一致

**核心问题**: AMD HIP/ROCm 上,RDMA 通过 `ibv_reg_mr()` fallback 直接写 GPU VRAM,
但内核 5.4 缺 `CONFIG_PCI_P2PDMA`,无 amdgpu peermem 驱动,
RDMA 写操作与 GPU L2 cache **不自动一致**。

**机制**:
1. Prefill 完成 KV 计算,通过 RDMA GDR 写入 decode pod 的 GPU VRAM
2. Prefill 发送 ZMQ "Success" 消息给 decode
3. Decode 收到 "Success" 后开始 forward pass
4. **问题**: ZMQ 消息到达时,GPU L2 cache 中的 stale cache lines 尚未失效
5. Decode forward pass 读到部分 stale KV 数据 → 乱码

**为什么 warm cache 正常**: Warm cache 时 KV 已在 GPU VRAM 中 (之前请求写入),
L2 cache 已一致,无需 RDMA 写入。

**`torch.cuda.synchronize()` 不够**: 在 `MooncakeKVReceiver.poll()` 加 sync 后,
200K cold-cache 仍然 0/5。`synchronize()` 等待 GPU compute 完成,
但不失效 L2 cache lines for externally-written (RDMA) data。

### 2.3 修复方案: Host Staging

**原理**: RDMA 写到 host RAM (CPU 一致性,无 GPU L2 问题) →
`hipMemcpy` H2D 复制到 GPU VRAM (正规 GPU 内存操作,保证一致性)。

**配置**: 两侧都要 `SGLANG_PD_HOST_STAGING=1`。
Decode-only 只能修 RDMA 写入 decode VRAM 后的 L2；prefill 刚算出的 KV
被 RDMA 从 GPU 读走时同样可能是 stale，必须先 D2H 再 RDMA。

## 3. 修复实施与测试结果

### 3.1 Patch 架构

容器镜像中的 conn.py (v0.5.17 base, 2451 行) 没有 host staging 代码。
本地 repo 的 conn.py (2171 行) 有完整 host staging 实现。
通过运行时 patch 脚本 `patch_host_staging.py` 注入:

1. `register_buffer_to_engine()`: 用 `hipMallocHost` 分配 host buffer,
   注册 host buffer 到 RDMA engine (替代 GPU buffer)
2. `_copy_host_to_gpu()`: `hipMemcpy` H2D 从 host buffer 复制到 GPU VRAM
3. `MooncakeKVReceiver.poll()`: 在 KV transfer Success 时调用 `_copy_host_to_gpu()`

### 3.2 Benchmark 结果

| 测试场景 | 时间 | Prefill | Decode | Facts | 状态 |
|----------|------|---------|--------|-------|------|
| 200K cold (无 host staging) | 183s | - | - | 0/5 | 乱码 |
| 200K cold (host staging v1) | **17.1s** | 9259 tok/s | 30 tok/s | **5/5** | ✅ 正确 |
| 200K warm (host staging v1) | - | - | - | HTTP 503 | RDMA context 死亡 |
| 32K cold (host staging v1) | 9.7s | 2618 tok/s | 37.6 tok/s | **5/5** | ✅ 正确 |
| 64K cold (host staging v1) | - | - | - | HTTP 500 | RDMA context 死亡 |

**结论**: Cold-cache 乱码修复**成功验证**!

### 3.3 遗留问题: RDMA Context 死亡

**现象**: 第一次请求 (cold) 成功 5/5,但 `_copy_host_to_gpu` 执行后,
所有 8 个 bnxt_re bond 报 `local access violation`,后续请求全部 HTTP 500/503。

**时间线**:
1. 12:02 Host staging 79 个 buffer 注册 (66.7 GB)
2. 13:02:27 32K 传输成功,`_copy_host_to_gpu` 复制全部 79 个 buffer (66.7 GB)
3. 13:02:38 (11 秒后) 所有 bnxt_re bond 报 `local access violation`
4. 后续所有 RDMA 传输失败,prefill circuit breaker 打开

**根因**: `hipMemcpy` DMA 复制 66.7 GB 数据时,
干扰了 bnxt_re RDMA NIC 的 IOMMU 映射 → QP 进入 error state。
Re-registration 内存 (`batch_deregister` + `batch_register`) 不能修复 QP 错误。

## 4. 后续排查方向

### 4.1 当前方案: 选择性复制 (v2 patch, **已验证通过**)

**思路**: 只复制实际 RDMA 传输的 KV pages (~几 GB),而非全部 66.7 GB pool。

**实现** (`patch_host_staging_v2.py`):
1. `send_metadata()` 存储 `kv_indices` (目标 KV page indices)
2. `poll()` 将 `kv_indices` 传给 `_copy_host_to_gpu(kv_indices)`
3. `_copy_host_to_gpu()` 按 kv_indices 计算偏移量,只复制传输的 pages
4. 用 contiguous index grouping 减少 hipMemcpy 调用次数

**偏移量计算**:
```
offset = start_idx * page_size * kv_item_lens[buf_idx]
length = count * page_size * kv_item_lens[buf_idx]
```

**验证结果 (2026-08-15 12:02–12:04, decode pod 重启后)**:

| 测试 | Time | Prompt tokens | Copy size | Facts | QP |
|------|------|---------------|-----------|-------|----|
| 200K iter1 cold | 10.7s | 158607 | 7.22 GB / 158607 pages / 1 group | **5/5** | 存活 |
| 200K iter2 warm | 9.6s | 158607 | 7.22 GB | **5/5** | 存活 |
| 200K iter3 warm | 9.8s | 158607 | 7.22 GB | **5/5** | 存活 |
| 32K-cold | 5.2s | 25469 | 1.16 GB | **5/5** | 存活 |
| 64K-cold | 6.6s | 50845 | 2.31 GB | **5/5** | 存活 |
| 100K-cold | 5.5s | 79393 | 3.61 GB | **5/5** | 存活 |
| 158K-cold | 6.4s | 125267 | 5.70 GB | **5/5** | 存活 |

日志证据:
- 仅出现 `_copy_host_to_gpu: selective copy ...`, **0 次 full copy**
- decode/prefill **无** `local access violation` / QP error / circuit breaker
- decode pod 保持 Ready, restarts=0

**结论**: 选择性复制把 H2D 从 66.7 GB 降到实际 KV 大小 (1–7 GB),修复了 v1 的 QP 死亡,同时保留 cold-cache 正确性。

### 4.2 备选方案 (如果 v2 失败)

#### 方案 A: CPU memcpy 中转
```
host_staging_buffer → cpu_memcpy → temp_host_buffer → hipMemcpy → GPU VRAM
```
避免 DMA 直接读 staging buffer,但多一次 66.7 GB host→host 拷贝,较慢。

#### 方案 B: 放弃 host staging,修 L2 cache
不用 host staging,改为在 RDMA 写 GPU VRAM 后:
- Launch 一个读 KV cache 地址的 dummy CUDA kernel (强制 L2 cache 刷新)
- 或用 `__amdgcn_buffer_wbinvl1` intrinsic (AMD GCN L1 cache invalidation)

**风险**: 需要验证 AMD HIP L2 cache 行为,可能需要自定义 CUDA kernel。

#### 方案 C: 升级内核启用 dmabuf
内核 5.4 缺 `CONFIG_PCI_P2PDMA` / `CONFIG_DMABUF_MOVE_NOTIFY`。
升级到 5.12+ 可启用 HIP dmabuf RDMA,绕过 `ibv_reg_mr()` fallback。
但 TKE 节点内核升级需要运维配合,风险高。

### 4.3 验证清单

- [x] v2/v3 decode 选择性复制: 200K×3 + 32K–158K **在 prefill cache hit 时** 5/5, QP 存活
- [x] v3: 空 `kv_indices` 不再回退全量 66.7GB 拷贝
- [x] 时间戳 needle 的 **真正 prefill-cold** 64K/200K（仅 decode staging）: 仍乱码 (`1.1.2...</think>` 循环)
- [x] 根因修正: decode host staging 修不了 prefill 刚写入、RDMA 从 GPU 读到的 stale KV
- [x] 整文件覆盖 `conn_v0516_fixed.py` 到 v0.5.17 **失败** (`send()` 缺 `num_kv_tokens`) — 已回滚
- [x] 定点 `patch_prefill_d2h.py` + 两侧 `HOST_STAGING=1`: unique 64K/200K 5/5，无乱码
- [x] unique 200K×3 + 并发 32K×2 全 PASS；之后短请求 `3+3` HTTP 200；0 QP error
- [x] Dockerfile / `patches/post1/conn_host_staging.py` 已接入 bake-in（尚未重建并滚动镜像）
- [x] `register_buffer_to_engine` 只在 decode 替换 `kv_data_ptrs`（prefill 保持 GPU 指针）
- [ ] 重建镜像 `sglang-glm52-308x:v0517-...` 并滚动 1P1D，去掉运行时 conn 补丁依赖

### 4.4 长期优化

1. **重建 Docker 镜像**: 将 host staging 代码 baked-in 到 conn.py,
   不再依赖运行时 patch
2. **上游 PR**: 将选择性 `_copy_host_to_gpu` 贡献到 sglang 主仓库
3. **内核升级**: 推动运维升级 TKE 节点内核到 5.12+,
   启用 HIP dmabuf RDMA (根本解决 L2 cache 一致性)

## 5. 关键文件与配置

### 5.1 Patch 文件 (decode pod `/data/mooncake-patched/`)

| 文件 | 作用 |
|------|------|
| `patch_host_staging.py` | 注入 host staging 代码到 conn.py (v1,全量复制) |
| `patch_host_staging_v2.py` | 选择性复制版本 (只复制传输的 KV pages) |
| `patch_rdma_hip_sync.py` | poll() 中加 `torch.cuda.synchronize()` (保留,但不够) |
| `patch_overlap_hip_wait.py` | HIP `publish_ready.synchronize()` → `wait()` |
| `patch_decode_pd_health_flush.py` | decode loop flush synthetic health reply |
| `patch_abort_noblock.py` | zmq HWM deadlock fix for abort |
| `dsa_indexer_v0517_fixed.py` | DSA fused-store length guard |
| `engine.cpython-310-x86_64-linux-gnu.so` | A-group engine.so (HIP dmabuf support) |
| `patch_prefill_d2h.py` | prefill `_transfer_data`: 每块 hipMallocHost + D2H 后再 RDMA（不改 `send()`） |

### 5.2 关键环境变量

| 环境变量 | decode | prefill | 说明 |
|----------|--------|---------|------|
| `SGLANG_PD_HOST_STAGING` | `1` | `1` | prefill D2H 后 RDMA；decode RDMA 写 host 再选择性 H2D |
| `SGLANG_DSA_HIP_DISABLE_PRESHUFFLE` | `1` | `1` | 必须保留,否则所有上下文乱码 |

### 5.3 本地 Patch 源文件

| 本地路径 | 说明 |
|----------|------|
| `docker/rocm-mi308x-glm52-pd/live-1p1d/patches/patch_host_staging.py` | v1 patch (全量复制, 由 v2 覆盖) |
| `docker/rocm-mi308x-glm52-pd/live-1p1d/patches/patch_host_staging_v2.py` | v2 patch (选择性复制, 已验证) |
| `docker/rocm-mi308x-glm52-pd/live-1p1d/patches/patch_prefill_d2h.py` | prefill D2H 定点补丁（已验证 unique-cold） |
| `docker/rocm-mi308x-glm52-pd/patches/post1/conn_host_staging.py` | 镜像构建时串联 v1/v2/D2H 三个补丁 |
| `python/sglang/srt/disaggregation/mooncake/conn.py` | 本地 repo：decode-only 替换 ptrs + prefill D2H + 选择性 H2D |

### 5.4 Benchmark 脚本

| 文件 | 说明 |
|------|------|
| `/data/bench_200k.py` (prefill pod) | 200K token, 5 needles, 3 iterations |
| `/data/bench_multictx.py` (prefill pod) | 32K/64K/100K/158K cold-cache 测试 |
| `/data/bench_unique_cold.py` (prefill pod) | 时间戳 needle，200K×3 + 并发 32K |
| `/data/bench_print_unique64k.py` / `200k.py` | 打印原文，确认不是 `1.1.2...</think>` 乱码 |

## 6. 关键经验教训

1. **AMD HIP RDMA L2 cache 不一致**: RDMA 写 GPU VRAM 后,GPU L2 cache 不会自动失效。
   `torch.cuda.synchronize()` 不够,需要 `hipMemcpy` 或 kernel launch 强制刷新。

2. **hipMemcpy DMA 杀死 RDMA NIC QP**: 大量 (66.7 GB) hipMemcpy DMA 干扰
   bnxt_re NIC 的 IOMMU 映射,导致 QP error。需要选择性复制。

3. **Host staging 是有效 workaround**: RDMA 写 host RAM (CPU 一致性) →
   hipMemcpy H2D (GPU 内存操作) 提供 memory barrier,修复 cold-cache 乱码。

4. **运行时 patch 的局限性**: 容器镜像 conn.py 与本地 repo 版本不同,
   需要运行时 patch 注入代码。长期应重建镜像 baked-in 修复。
   **不要**把带旧 `send()` 签名的整份 `conn_v0516_fixed.py` 盖到 v0.5.17 上。

5. **Prefill 侧同样有 L2 不一致**: decode-only host staging 在 prefill radix
   命中时看起来像 5/5，换时间戳 needle 强迫重算后仍乱码。必须 D2H 后再 RDMA。
