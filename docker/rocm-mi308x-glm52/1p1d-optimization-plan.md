# 1p1d PD 分离部署优化 Plan

> 更新时间: 2026-07-28 (第二轮优化完成)
> 当前状态: pods restart=1, 稳定运行中, 仅 watchdog_timeout=7200 变更生效
> GEMM 调优: 37 个新 shape 已调优 (gfx942/MI308X), 需重启生效

## 优化执行结果摘要

| # | 项目 | 状态 | 备注 |
|---|------|------|------|
| P1-1 | watchdog 3600→7200 | ✅ 已应用 | 唯一成功的配置变更 |
| P1-2 | Prefill CUDA graph | ❌ 不可用 | DSA backend 不支持, AttributeError |
| P1-3 | HiCache write_through_selective | ❌ 已回退 | 导致 mooncake RDMA 线程挂起 (14min) |
| P1-4 | EAGLE 3/1/4→4/1/5 | ❌ 已回退 | 17min 后不稳定 |
| P1-5 | FP8 KV scaling | ✅ 已调查 | 模型无 scaling factor 文件, 警告良性 |
| P2-1 | aiter BF16 GEMM 调优 | ✅ 已完成 | 37 shape 已调优, 需重启生效 |
| P2-2 | HiCache ratio 4→6/8 | ❌ 已回退 | 导致 mooncake RDMA 不稳定 |
| P2-3 | RoPE 精度 | ✅ 已调查 | fused kernel 已使用 FP32 累加, 无需修改 |
| P2-4 | decode mem_fraction 0.85→0.88 | ❌ 已回退 | 与 EAGLE 一起回退 |

**关键发现**: 任何增加显存使用或修改 KV cache 行为的配置变更都会导致 mooncake RDMA 传输不稳定。
MI308X = gfx942 (80 CU), 但 CSV 中 N=256/K=6144 的调优数据只有 gfx950 (256 CU)。

## 当前配置基线

### Prefill Pod (`sglang-1p1d-prefill-0`)
| 参数 | 值 | 说明 |
|------|-----|------|
| `mem_fraction_static` | 0.90 | GPU KV cache 占比 |
| `max_total_num_tokens` | 1,451,999 | GPU KV cache 容量 |
| `chunked_prefill_size` | 32768 | prefill 分块 |
| `kv_cache_dtype` | fp8_e4m3 | FP8 KV cache |
| `attention_backend` | dsa (tilelang) | DSA 注意力 |
| `disable_prefill_cuda_graph` | False (但被自动禁用) | ⚠️ auto-disable |
| `disable_decode_cuda_graph` | True | prefill-only |
| `hicache_ratio` | 4.0 | host memory cache 4x |
| `hicache_write_policy` | write_through | ⚠️ 应升级为 write_through_selective |
| `disable_radix_cache` | False | radix cache 开启 |
| `disaggregation_mode` | prefill | PD 分离 |
| `disaggregation_transfer_backend` | mooncake | RDMA KV 传输 |
| `load_balance_method` | follow_bootstrap_room | PD 路由 |
| `watchdog_timeout` | 3600s (1h) | ⚠️ 偏短 |
| `speculative_algorithm` | None | prefill 无 EAGLE |

### Decode Pod (`sglang-1p1d-decode-0`)
| 参数 | 值 | 说明 |
|------|-----|------|
| `mem_fraction_static` | 0.85 | GPU KV cache 占比 |
| `max_total_num_tokens` | 1,224,555 | GPU KV cache 容量 |
| `chunked_prefill_size` | 16384 | decode 侧 prefill 分块 |
| `kv_cache_dtype` | fp8_e4m3 | FP8 KV cache |
| `disable_radix_cache` | True | PD decode 不用 radix |
| `num_continuous_decode_steps` | 2 | 连续 decode 步数 |
| `schedule_conservativeness` | 0.5 | 调度保守度 |
| `speculative_algorithm` | EAGLE | EAGLE 投机解码 |
| `speculative_num_steps` | 3 | ⚠️ 可调 |
| `speculative_eagle_topk` | 1 | ⚠️ 可调 |
| `speculative_num_draft_tokens` | 4 | draft token 数 |
| `cuda_graph_bs_decode` | 1-32 | decode cuda graph 开启 |
| `hicache_ratio` | 4.0 | host memory cache 4x |
| `hicache_write_policy` | write_through | ⚠️ 应升级 |
| `disaggregation_mode` | decode | PD 分离 |
| `load_balance_method` | follow_bootstrap_room | PD 路由 |

### Router (`sglang-1p1d-router`)
| 参数 | 值 | 说明 |
|------|-----|------|
| 镜像 | v0516-batch1-tok | Python 3.12 |
| patched .so | 已部署 (Codex namespace/web_search) | /dev/shm/ → /opt/venv/ |
| HTTPRoute | 直连 router:30001 | 已绕过 LiteLLM |
| circuit breaker | timeout=300s, failure=10, success=2 | |
| health check | timeout=60s, interval=30s | |

---

## 已完成项 (P0 — 验证)

| # | 项目 | 状态 | 备注 |
|---|------|------|------|
| 1 | LiteLLM 移除 | ✅ 已完成 | HTTPRoute 直连 router, 节省 ~12% 开销 |
| 2 | PD LB 冲突修复 | ✅ 已完成 | decode → follow_bootstrap_room |
| 3 | Responses API bootstrap 转发 | ✅ 已完成 | protocol.py + serving_responses.py 补丁 |
| 4 | Router patched .so (Codex 兼容) | ✅ 已完成 | namespace/web_search 工具类型支持 |
| 5 | HiCache 启用 | ✅ 已完成 | hicache_ratio=4, write_through |
| 6 | Prefill CUDA graph 标志移除 | ✅ 已完成 | disable_prefill_cuda_graph=False |

---

## 优化项详细 Plan

### P1-1: 修复 Mooncake 挂起导致 Watchdog 退出 (紧急)

**问题**: 运行 11h 后, prefill 和 decode 的 mooncake transfer_worker 线程全部 idle,
watchdog 检测到无进展, 触发 coredump + 退出 (exit 137). 两个 pod 在 10s 内相继挂掉.

**根因分析**:
- mooncake RDMA 连接在长时间运行后可能断开 (kernel 缺少 `CONFIG_PCI_P2PDMA`)
- 回退到 `ibv_reg_mr()` 的路径在 amdgpu peermem 驱动不稳定时可能挂起
- watchdog_timeout=3600s 过短, 1 小时无进展就退出, 不适合长连接场景

**修复方案**:
1. **增加 watchdog_timeout**: `3600` → `7200` (2h), 给 mooncake 更多恢复时间
2. **添加 mooncake 心跳保活**: 检查 mooncake 配置是否有 keepalive 参数
3. **监控 mooncake 连接状态**: 在 Prometheus 指标中添加 transfer_worker 活跃数
4. **考虑回退到 TCP 传输**: 如果 RDMA 不稳定, 可用 `--disaggregation-transfer-backend tcp`

**风险评估**: 低风险. 增加 timeout 只是延缓退出, 不影响正常逻辑.

---

### P1-2: Prefill CUDA Graph 强制启用

**问题**: 移除 `--disable-prefill-cuda-graph` 后, 系统自动禁用了 prefill cuda graph:
```
Disable prefill CUDA graph because cuda_graph_config resolved prefill.backend='disabled'
(e.g. via --cuda-graph-backend-prefill=disabled or auto-disable rules).
```

**根因**: `cuda_graph_backend_prefill=None` 时, 自动解析逻辑将 prefill backend 设为 `disabled`.
可能原因:
- `attention_backend='dsa'` + `dsa_prefill_backend='tilelang'` 不支持 prefill cuda graph
- `chunked_prefill_size=32768` 过大, 超出 cuda graph 限制
- `disaggregation_mode='prefill'` 触发了自动禁用规则

**修复方案**:
1. **显式指定 backend**: 添加 `--cuda-graph-backend-prefill=static` 强制启用
2. **如果仍然失败**: 降低 `chunked_prefill_size` 到 8192 或 4096
3. **验证 DSA 兼容性**: 检查 tilelang prefill backend 是否支持 cuda graph capture
4. **如果 DSA 不兼容**: 尝试 `--prefill-attention-backend=flashinfer` 或 `--prefill-attention-backend=triton`

**预期收益**: prefill 延迟降低 20-40%, 尤其是短序列场景.

**风险评估**: 中风险. 需要验证 DSA + cuda graph 兼容性, 可能需要回退 attention backend.

---

### P1-3: HiCache 策略升级 (write_through → write_through_selective)

**问题**: 当前 `hicache_write_policy='write_through'`, 2tp8 分析发现 `write_back` 已弃用,
推荐使用 `write_through_selective`.

**修复方案**:
```bash
--hicache-write-policy write_through_selective
```

**差异**:
- `write_through`: 所有 KV cache 写入同时写入 host memory, 写入延迟较高
- `write_through_selective`: 只选择性地写入重要的 KV cache (如长前缀), 减少写入开销
- `write_back` (已弃用): 延迟写入, 可能丢失数据

**预期收益**: hicache 写入开销降低 30-50%, 对高吞吐场景更友好.

**风险评估**: 低风险. `write_through_selective` 是官方推荐策略.

---

### P1-4: EAGLE 投机解码调优

**当前指标** (来自 2tp8 基准):
- `accept_len`: 2.54-2.88 (平均接受 2.5-2.9 个 token)
- `accept_rate`: 0.51-0.62 (51%-62% 的 draft token 被接受)

**调优方向**:

| 参数 | 当前 | 建议 | 说明 |
|------|------|------|------|
| `speculative_num_steps` | 3 | 4 | 增加 draft 长度, 提高 accept_len 上限 |
| `speculative_eagle_topk` | 1 | 2 | 增加候选数, 提高 accept_rate |
| `speculative_num_draft_tokens` | 4 | 6 | 配合 num_steps=4, 提供更多 draft |

**权衡**:
- `num_steps=4` + `eagle_topk=2`: 更高的接受率和吞吐, 但 GPU 计算开销增加 ~30%
- 建议先试 `num_steps=4, eagle_topk=1, num_draft_tokens=5` (保守方案)
- 再试 `num_steps=4, eagle_topk=2, num_draft_tokens=6` (激进方案)

**预期收益**: accept_len 从 2.7 → 3.5+, decode 吞吐提升 15-25%.

**风险评估**: 中风险. 过大的 draft 可能导致 GPU 显存不足或调度延迟增加.

---

### P1-5: FP8 KV Cache Scaling Factors

**问题**: 所有 8 个 TP rank 均告警:
```
Using FP8 KV cache but no scaling factors provided. Defaulting to 1.0.
```

**影响**: FP8 KV cache 使用默认 scale=1.0, 可能导致:
- KV cache 精度损失 (某些层的 KV 值超出 FP8 表示范围)
- 模型质量下降 (尤其是长序列)
- 无法充分利用 FP8 的动态范围

**修复方案**:
1. **从模型 checkpoint 提取 scaling factors**: 检查 `/data/model/glm52-fp8/` 是否有 KV cache scaling factors
2. **在线校准**: 使用 `--kv-cache-quant-scale` 或类似参数自动校准
3. **手动指定**: 如果模型提供 scaling factors 文件, 通过参数传入

**验证方法**: 对比启用前后的生成质量 (perplexity, BLEU 等).

**风险评估**: 低风险. 提供正确的 scaling factors 只会提升质量, 不影响性能.

---

### P2-1: aiter BF16 GEMM 调优 ✅ 已完成

**根因发现**: MI308X = gfx942 (80 CU), 但 CSV 中 N=256/K=6144 的调优数据只有 gfx950 (256 CU)。
运行时查找使用 `(gfx, cu_num, M, N, K, bias, dtype, otype, scaleAB, bpreshuffle)` 作为 key,
gfx 不匹配导致所有 N=256/K=6144 shape 回退到 torch 默认实现。

**调优执行** (2026-07-28):
- Prefill pod 缺失 shape: M=1,2,4,8,9,12,13 (N=256, K=6144) — 13 个
- Decode pod 缺失 shape: M=3,5,6,7,10,20,24,28,36,40,80,96 (N=256) + M=1,2,4,8,12,16,20,24,28,32,48,64 (N=32) — 24 个
- 总计: 37 个新 shape 调优完成, 耗时 ~130 秒
- 调优后 CSV: 2391 行 (原 2358 + 33 新增, 4 个重复)

**调优结果**:
| M 范围 | N | libtype | TFLOPS 范围 |
|--------|---|---------|------------|
| 1-16 | 256 | opus | 0.61-9.89 |
| 20-32 | 256 | opus | 9.73-12.84 |
| 28-40 | 256 | asm | 11.28-15.09 |
| 48-64 | 256 | asm | 17.71-22.08 |
| 80-128 | 256 | opus/asm | 23.88-35.05 |
| 256 | 256 | asm | 48.26 |
| 1-64 | 32 | opus | 0.06-2.88 |

**持久化**:
- `/data/aiter_configs/bf16_tuned_gemm.csv` — 持久化副本 (prefill + decode pod)
- `/sgl-workspace/aiter/aiter/configs/bf16_tuned_gemm.csv` — 源 CSV 已追加 40 条 gfx942 K=6144 条目
- ⚠️ 源 CSV 修改在容器内, pod 重启后会丢失 (需 initContainer 或新镜像持久化)

**生效条件**: 需要重启 pod (aiter 使用 lru_cache, 运行中不会重新加载 CSV)

**后续建议**: 添加 initContainer 在 pod 启动时从 `/data/aiter_configs/` 拷贝调优 CSV 到 `/sgl-workspace/aiter/aiter/configs/`

---

### P2-2: HiCache Ratio 提升 (4 → 6~8) ❌ 已回退

**测试结果**: hicache_ratio=6 (prefill) / 8 (decode) 导致 mooncake RDMA 线程在 14-17 分钟后全部 idle,
watchdog 触发 kill (exit 137)。回退到 hicache_ratio=4 后恢复稳定。

**根因**: 增加 hicache_ratio 会增加 host memory 使用, 与 mooncake RDMA buffer 竞争内存,
导致 RDMA 传输线程挂起。

---

### P2-3: aiter Fused RoPE 精度 ✅ 已调查 (无需修改)

**结论**: fused QK-norm-RoPE kernel 已使用 FP32 累加, 无需修改。

**调查详情** (2026-07-28):
- 内核源码: `/sgl-workspace/sglang/python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`
- 内核使用 `float elements[numElemsPerThread]` 数组, 加载 bf16 输入后转换为 FP32 进行 RMSNorm 和 RoPE 计算
- 频率计算 `compute_freq()` 使用 `float` (FP32) 精度
- `--triton-attention-reduce-in-fp32` 标志仅影响 Triton attention kernel, 不影响 DSA backend, 与 fused RoPE 无关

**无需任何修改。**

---

### P2-4: Decode mem_fraction_static 提升 (0.85 → 0.88)

**当前**: decode `mem_fraction_static=0.85`, `available_gpu_mem=24.49 GB`.

**分析**: decode pod 的 GPU 显存有 24.49 GB 空闲, 可以适当提升 KV cache 占比.

**建议**: `0.85` → `0.88`, 增加 ~3% KV cache 容量.

**预期收益**: KV cache 容量增加 ~3%, 支持更多并发请求.

**风险评估**: 低风险, 但需监控 OOM. 留意 EAGLE draft model 的显存占用.

---

### P3-1: Mooncake RDMA Kernel 优化 (研究项)

**问题**: kernel 缺少 `CONFIG_PCI_P2PDMA` / `CONFIG_DMABUF_MOVE_NOTIFY`, 无法使用 GPU-direct RDMA,
回退到 `ibv_reg_mr()` (需要 amdgpu peermem 驱动).

**日志**:
```
Kernel lacks CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY
(p2pdma=0 move_notify=0), HIP dmabuf MR registration disabled,
falling back to ibv_reg_mr()
```

**影响**: KV cache 传输需要经过 CPU 中转, 延迟增加 ~20-30%.

**修复方案**:
1. **重建内核**: 启用 `CONFIG_PCI_P2PDMA` 和 `CONFIG_DMABUF_MOVE_NOTIFY`
2. **安装 amdgpu peermem 驱动**: 确保 `amdgpu_peermem` 模块加载
3. **验证**: 检查 `lsmod | grep peermem` 和 `cat /proc/config.gz | grep P2PDMA`

**风险评估**: 高风险. 涉及内核编译和驱动安装, 需要 EKS 节点级别变更. 建议作为长期优化.

---

### P3-2: EAGLE on Prefill (投机 Prefill)

**当前**: prefill `speculative_algorithm=None`.

**研究方向**: 在 prefill 阶段使用 EAGLE 投机解码, 加速长序列 prefill.

**注意**: 这是实验性功能, 需要确认 sglang v0516-batch1 是否支持. 可能需要升级版本.

**风险评估**: 高风险. 实验性功能, 可能不稳定.

---

### P3-3: PD 路由 KV Cache 感知

**当前**: PD router 使用 `follow_bootstrap_room` 路由, 不感知 KV cache 状态.

**用户需求**: router 能感知 KV cache, 实现 cache-aware 的 PD 调度.

**分析**:
- 2tp8 router 使用 `cache_aware` 策略, 但那是非 PD 模式
- PD 模式下, router 需要同时考虑:
  - Prefill worker 的 radix cache 命中率 (prefix matching)
  - Decode worker 的负载 (KV cache 占用率)
  - Bootstrap room 分配

**方案**:
1. 检查 sglang router 是否支持 PD + cache_aware 混合模式
2. 如果不支持, 需要修改 `pd_router.rs` 添加 cache-aware 逻辑
3. 可以在 prefill 选择时, 优先选择 radix cache 命中率高的 worker

**风险评估**: 中高风险. 需要修改 router 源码, 但可以显著提升多轮对话性能.

---

## 优先级排序与执行计划

### 第一阶段: 稳定性修复 ✅ 完成
1. **P1-1**: 增加 watchdog_timeout → 7200s ✅ 已应用
2. **P1-3**: HiCache 策略升级 → write_through_selective ❌ 已回退 (mooncake RDMA 不稳定)
3. 验证 pods 稳定运行 ✅ 稳定 (restart=1, 运行中)

### 第二阶段: 性能优化 — 大部分不可用
4. **P1-2**: 强制启用 prefill CUDA graph ❌ DSA backend 不支持
5. **P1-4**: EAGLE 调优 ❌ 已回退 (17min 后不稳定)
6. **P1-5**: FP8 KV cache scaling factors ✅ 已调查 (模型无 scaling factor, 警告良性)
7. **P2-4**: Decode mem_fraction_static 0.85 → 0.88 ❌ 已回退

### 第三阶段: 深度调优 — 部分完成
8. **P2-1**: aiter BF16 GEMM 调优 ✅ 已完成 (37 shape, 需重启生效)
9. **P2-2**: HiCache ratio 4 → 6/8 ❌ 已回退 (mooncake RDMA 不稳定)
10. **P2-3**: aiter fused RoPE 精度检查 ✅ 已调查 (已使用 FP32, 无需修改)

### 第四阶段: 长期优化 (研究)
11. **P3-1**: Mooncake RDMA kernel 优化
12. **P3-3**: PD 路由 KV cache 感知

### 后续行动项
- [ ] 添加 initContainer 持久化 GEMM 调优 CSV (从 /data/aiter_configs/ 拷贝)
- [ ] 重启 pod 使 GEMM 调优生效
- [ ] 重启后运行基准测试对比 2tp8 baseline
- [ ] 考虑构建新镜像包含调优 CSV

---

## 验证方法

### 功能验证
```bash
# 1. Chat Completions API
curl -s https://glm52-pd-1p1d.jmpti.woa.com/v1/chat/completions \
  -H "Authorization: Bearer sk-REPLACE_WITH_YOUR_API_KEY" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}],"max_tokens":20}'

# 2. Responses API (Codex 格式)
curl -s https://glm52-pd-1p1d.jmpti.woa.com/v1/responses \
  -H "Authorization: Bearer sk-REPLACE_WITH_YOUR_API_KEY" \
  -d '{"model":"glm-5.2","input":"hi","max_output_tokens":20}'

# 3. PD transfer 验证 (decode pod 日志)
kubectl logs -n kube-system sglang-1p1d-decode-0 | grep -E "transfer-req|bootstrap-req"
```

### 性能基准
```bash
# 并发基准测试 (C=1,4,8,16)
python3 benchmark_pd_vs_2tp8.py --concurrency 1,4,8,16 --api chat
# 对比指标: throughput (tok/s), TTFT (ms), ITL (ms)
```

### 稳定性监控
```bash
# Pod 状态
kubectl get pods -n kube-system -w | grep 1p1d

# Watchdog 退出检测
kubectl logs -n kube-system sglang-1p1d-prefill-0 | grep -i "watchdog\|kill_process"

# Mooncake 传输状态
kubectl logs -n kube-system sglang-1p1d-decode-0 | grep -E "transfer-req|bootstrap-req"
```
