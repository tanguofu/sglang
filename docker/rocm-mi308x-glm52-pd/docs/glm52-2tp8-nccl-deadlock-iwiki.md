# GLM-5.2 2tp8 NCCL Deadlock 修复完整记录

## 1. 问题概述

| 项目 | 内容 |
|------|------|
| **环境** | GLM-5.2 2tp8 SGLang 部署 (MI308X gfx942) |
| **镜像** | `xgmi-opt-0716d` (sglang b76dd0be69, 2026-07-10) |
| **现象** | 长 token 请求 (1K/3K/5K) 触发 NCCL BROADCAST 死锁,worker 被 watchdog kill |
| **根因** | PR #31478 hack 代码在 EAGLE greedy 分支添加 3 个 NCCL broadcast,rank 间 collective 不匹配 |
| **修复** | 切换镜像 `fix-eagle-coredump-v3` (sglang 48ae829f6e, 2026-07-20) |
| **验证** | 10/10 TTFT 测试通过 (含之前必崩的 3K iter=1 和 5K 场景) |
| **日期** | 2026-07-22 |

---

## 2. 现象描述

### 2.1 TTFT 异常 (非线性延迟)

通过 envoy gateway (`glm52-2tp8.jmpti.woa.com`) 测试不同输入长度:

| 测试场景 | iter=0 TTFT | iter=1 TTFT | iter=2 TTFT | 结果 |
|----------|-------------|-------------|-------------|------|
| warmup (small) | 10.5s | - | - | ✅ |
| 1K tokens | 23.7s | 0.24s | 0.24s | ✅ (首次冷启动慢) |
| 3K tokens | 4.6s | **TIMEOUT 90s** | **TIMEOUT 90s** | ❌ 第二次请求杀死 worker |
| 1K-warm | **TIMEOUT 90s** | **TIMEOUT 90s** | - | ❌ worker 已死 |
| 5K tokens | **TIMEOUT 120s** | - | - | ❌ |

**关键模式**: 3K 首次请求成功 (4.6s),第二次请求导致 worker 崩溃,后续所有请求超时。

### 2.2 Worker 崩溃日志

```
[WATCHDOG] Scheduled kill_process_tree due to no progress in 600057ms
WorkNCCL(SeqNum=25, OpType=BROADCAST, NumelIn=4, NumelOut=4)
  Rank 0: timeout 600000ms
  Rank 1: timeout 600000ms
  ...
  Rank 7: timeout 600000ms
```

- **SeqNum=25**: 第 25 个 NCCL collective 操作
- **NumelIn=4**: 1 个 int32 (4 bytes) — EAGLE verify 的 `bs=1 predict` 或 `num_correct_drafts`
- **OpType=BROADCAST**: 集体广播操作
- **600s 超时**: 所有 8 个 TP rank 同时阻塞

---

## 3. 根因分析

### 3.1 容器内 hack 代码审计

镜像 `xgmi-opt-0716d` 基于 sglang commit `b76dd0be69` (2026-07-10),包含大量 hack 代码:

| 文件 | 修改 | 风险 |
|------|------|------|
| `eagle_utils.py:683-685` | PR #31478: greedy 分支添加 3 个 NCCL broadcast | **高 — 死锁根因** |
| `common.py:2137` | `force_cpu_device: True → False` | **高 — 让 broadcast_pyobj 走 GPU** |
| `request_receiver.py` | `device_group → cpu_group` (apply_eagle_patch.py) | 中 — 运行时 patch |
| `dsa_indexer.py` | 多处 `_is_cuda → _is_cuda or _is_hip` | 低 — HIP 兼容性 |
| `dsa_backend.py` | `assert → graceful None`, `.view() → .reshape()` | 低 — 容错性 |
| `eagle_worker_v2.py:466` | DSA draft extend graph 添加 `_is_hip` | 低 — HIP 兼容性 |
| `radix_attention.py` | 禁用 `_pcg_mha_companion` | 低 — 功能开关 |
| `deepseek_v2.py` | PCG dual stream 添加 `_is_hip` | 低 — HIP 兼容性 |
| `deepseek_nextn.py` | alt_stream 添加 `_is_hip` | 低 — HIP 兼容性 |
| `breakable_cuda_graph.py` | tuple/list 处理 weak_ref/copy_output | 低 — 容错性 |
| `fp8.py` | is_shuffled tracking | 低 — 功能增强 |
| `detokenizer_manager.py` | `poll(timeout=1000)` | 低 — 超时调整 |

### 3.2 死锁根因: PR #31478 Hack

#### 上游正确逻辑

EAGLE verify 阶段,greedy 分支 (HIP/ROCm 使用 `torch.argmax`,确定性) **不需要** NCCL broadcast:
- 上游 `eagle_utils.py` greedy 分支只有本地 `argmax` 操作
- PR #31478 的正确做法是在 `SIMULATE_ACC_LEN` 块后 **统一 hoist 单个 broadcast**
- 只在 sampling 分支需要 broadcast (因为 sampling 有随机性,各 rank 结果不同)

#### Hack 代码的错误

Hack 在 greedy 分支添加了 3 个 `tp_group.broadcast()`:

```python
# eagle_utils.py:683-685 (hack 代码)
tp_group.broadcast(src=bs_predict, dst=bs_predict)       # broadcast #1
tp_group.broadcast(src=num_correct_drafts, dst=num_correct_drafts)  # broadcast #2
# ... (第 3 个 broadcast)
```

**关键**: `tp_group.broadcast()` 使用 `device_group` (NCCL 后端),不是 `cpu_group` (Gloo)。

#### 死锁触发条件

当某些 rank 进入 idle 路径 (`batch.forward_mode.is_idle()`) 而其他 rank 进入 verify 路径时:
- Idle rank: **不调用** broadcast
- Verify rank: **调用** broadcast
- 结果: NCCL collective 不匹配 → 所有 rank 阻塞 → watchdog 600s 超时 → kill_process_tree

### 3.3 次要问题: force_cpu_device=False

`common.py:2137` 将 `broadcast_pyobj` 的 `force_cpu_device` 从 `True` 改为 `False`:
- **上游默认 `True`**: 创建 CPU tensor + 使用 Gloo cpu_group
- **Hack 改为 `False`**: 创建 GPU tensor + 使用 Gloo cpu_group (ROCm 上脆弱)

正确逻辑: `force_cpu_device=True` (上游默认),避免在 ROCm 上混合 GPU tensor 和 Gloo CPU group。

---

## 4. 镜像对比

| 项目 | xgmi-opt-0716d (旧) | fix-eagle-coredump-v3 (新) |
|------|---------------------|---------------------------|
| sglang commit | b76dd0be69 (2026-07-10) | 48ae829f6e (2026-07-20) |
| commit 差距 | - | +380 commits |
| PR #31478 | 运行时 hack (apply_eagle_patch.py) | 构建时合入 (正确实现) |
| force_cpu_device | False (hack) | True (上游默认) |
| apply_eagle_patch.py | 需要 (运行时 patch) | 不需要 (已内置) |
| EAGLE greedy broadcast | 3 个 NCCL broadcast (死锁) | 无 (正确) |
| ROCm | 7.2.0 | 7.2.4 |
| amdgpu | 6.16.13 | 6.16.13 |
| PyTorch | 2.9.1+rocm7.2.0 | 2.9.1+rocm7.2.0 |
| sgl-kernel | 0.4.4 | 0.4.4 |

---

## 5. 修复方案

### 方案选择: 切换镜像 (方案 A)

| 方案 | 描述 | 风险 | 选择 |
|------|------|------|------|
| A. 切换镜像 | 使用 `fix-eagle-coredump-v3` | 低 — 已在 prof19 验证 2 天 | ✅ |
| B. 手动修复 hack | 修改容器内代码 | 中 — 需重新构建镜像 | ❌ |

### 5.1 STS Patch 内容

通过 `kubectl patch` 直接修改 StatefulSet (后续同步到 helm values):

**变更 1: 镜像切换**
```
xgmi-opt-0716d → fix-eagle-coredump-v3
```

**变更 2: 移除 apply_eagle_patch.py**
```diff
- python3 /tmp/apply_eagle_patch.py || echo "WARN: patch failed, continuing"
```
(PR #31478 已在 `fix-eagle-coredump-v3` 构建时合入,无需运行时 patch)

**变更 3: cuda-graph-backend-prefill**
```diff
- --cuda-graph-backend-prefill breakable
+ --cuda-graph-backend-prefill tc_piecewise
```
(ROCm 不支持 BCG — breakable cuda graph 仅在 CUDA 上验证过,ROCm 必须用 `tc_piecewise`)

**变更 4: hicache-ratio**
```diff
- --hicache-ratio 4
+ --hicache-ratio 2
```
(ratio=4 导致 DSA indexer host memory ~2.4TB > 节点 RAM ~2TB,降为 2 安全)

### 5.2 保留的配置 (2tp8 并发优化)

以下配置在 2026-07-21 合并 chart 时已优化,保持不变:
- `--prefill-max-requests 8` (从 32 降)
- `--max-running-requests 48` (从 32 升)
- `--cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 12 16 20 24 32` (扩展到 32)
- `--cuda-graph-max-bs-decode 32` (从 16 升)
- `--mem-fraction-static 0.75` (从 0.82 降,OOM 修复)
- `--schedule-conservativeness 1` (从 0.5 升)
- `--chunked-prefill-size 16384` (从 131072 降)
- `--watchdog-timeout 1200` (从 3600 降)
- `--hicache-write-policy write_back` (从 write_through 改)
- `ROCM_QUICK_REDUCE_QUANTIZATION=NONE` (从 INT8 改,零精度损失)

---

## 6. 验证结果

### 6.1 prof19 预验证 (fix-eagle-coredump-v3)

在 node 21.234.170.19 的 `prof19-sglang-0` (已运行 fix-eagle-coredump-v3 2 天,0 restarts) 上测试:

| 测试 | TTFT | 结果 |
|------|------|------|
| warmup | 1.395s | ✅ |
| 1K iter=0 | 24.0s | ✅ |
| 1K iter=1 | 0.37s | ✅ |
| 1K iter=2 | 0.37s | ✅ |
| 3K iter=0 | 3.96s | ✅ |
| 3K iter=1 | 0.63s | ✅ |
| 3K iter=2 | 0.45s | ✅ |
| 1K-warm iter=0 | 0.37s | ✅ |
| 1K-warm iter=1 | 0.38s | ✅ |
| 5K | 1.17s | ✅ |

**10/10 全部通过**,包括之前必崩的 3K iter=1 场景。

### 6.2 2tp8 修复后 gateway 验证

通过 envoy gateway (`glm52-2tp8.jmpti.woa.com`) 测试:

| 测试 | TTFT | 结果 |
|------|------|------|
| warmup | 11.5s | ✅ |
| 1K iter=0 | 34.5s | ✅ |
| 1K iter=1 | 0.65s | ✅ |
| 1K iter=2 | 0.67s | ✅ |
| 3K iter=0 | 4.42s | ✅ |
| 3K iter=1 | 0.90s | ✅ |
| 3K iter=2 | 0.71s | ✅ |
| 1K-warm iter=0 | 0.60s | ✅ |
| 1K-warm iter=1 | 0.63s | ✅ |
| 5K | 1.42s | ✅ |

**10/10 全部通过**。3K 第二次请求不再崩溃,5K 正常响应。

### 6.3 Pod 状态

```
NAME                          READY   STATUS    RESTARTS   AGE
sglang-glm52-2tp8-sglang-0   1/1     Running   0          30m
sglang-glm52-2tp8-sglang-1   1/1     Running   0          25m
```

两个 worker (W1: .152, W2: .172) 均正常运行,0 restarts。

---

## 7. Helm Values 同步

STS patch 通过 `kubectl patch` 直接应用,后续同步到 helm chart values 文件:

**文件**: `deployments/glm52-tp8-0718/configs/pd-manifests/sglang-glm52-2tp8-values.yaml`

**4 处变更**:

```yaml
# 1. 镜像 tag
-tag: xgmi-opt-0716d
+tag: fix-eagle-coredump-v3

# 2. EAGLE patch 禁用 (已内置)
eaglePatch:
-  enabled: true
+  enabled: false

# 3. cuda-graph-backend-prefill
sglang:
-  cudaGraphBackendPrefill: breakable
+  cudaGraphBackendPrefill: tc_piecewise

# 4. hicache-ratio
-  hicacheRatio: 4
+  hicacheRatio: 2
```

---

## 8. 故障排查手册

### 8.1 如何识别 NCCL 死锁

```bash
# 查看 worker 日志
kubectl logs -n kube-system sglang-glm52-2tp8-sglang-0 --tail=200 | grep -E "WATCHDOG|NCCL|BROADCAST|timeout"

# 关键标志
# - "Scheduled kill_process_tree due to no progress in 600057ms"
# - "WorkNCCL(SeqNum=XX, OpType=BROADCAST, NumelIn=4)"
# - 所有 rank 同时 timeout
```

### 8.2 如何验证修复

```bash
# 1. 确认镜像
kubectl get sts -n kube-system sglang-glm52-2tp8-sglang \
  -o jsonpath='{.spec.template.spec.containers[0].image}'
# 期望: mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3

# 2. 确认无 apply_eagle_patch.py
kubectl get sts -n kube-system sglang-glm52-2tp8-sglang \
  -o jsonpath='{.spec.template.spec.containers[0].args}' | grep apply_eagle_patch
# 期望: 无输出

# 3. TTFT 测试 (通过 gateway)
for tokens in 1000 3000 5000; do
  curl -sk https://glm52-2tp8.jmpti.woa.com/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d "{\"model\":\"glm-5.2\",\"messages\":[{\"role\":\"user\",\"content\":\"$(python3 -c "print('x '*$tokens)")\"}],\"max_tokens\":10,\"stream\":false}" \
    -w "\n${tokens} tokens: TTFT=%{time_starttransfer}s\n" -o /dev/null
done
```

### 8.3 回滚方案

如需回滚到 `xgmi-opt-0716d` (不推荐,会重新引入死锁):

```bash
kubectl patch statefulset -n kube-system sglang-glm52-2tp8-sglang \
  --type='json' \
  -p='[{"op":"replace","path":"/spec/template/spec/containers/0/image","value":"mirrors.tencent.com/ti-platform/sglang-glm52-308x:xgmi-opt-0716d"}]'
```

---

## 9. 相关文档

- [GLM-5.2 MI308X EAGLE Coredump 修复完整部署文档](https://iwiki.woa.com/p/4026586166) — iWiki docid 4026586166
- sglang PR #31478: TP broadcast for EAGLE verify results
- Helm values: `deployments/glm52-tp8-0718/configs/pd-manifests/sglang-glm52-2tp8-values.yaml`
- STS backup: `/tmp/2tp8-fix-backup/sts-pre-fix.yaml`

---

## 10. 经验总结

1. **Hack 代码风险**: 运行时 patch (`apply_eagle_patch.py`) 修改 NCCL collective 行为,极易在 idle/verify 分支不一致时触发死锁。应优先使用构建时合入的官方修复。

2. **EAGLE collective 一致性**: EAGLE verify 的 greedy/sampling 分支必须保持 NCCL collective 一致性。greedy 分支 (确定性) 不需要 broadcast,sampling 分支 (随机性) 才需要。

3. **ROCm 限制**: `breakable` cuda graph 仅在 CUDA 上验证,ROCm 必须用 `tc_piecewise`。`force_cpu_device=False` 在 ROCm 上脆弱,应保持上游默认 `True`。

4. **helm drift 预防**: 直接 kubectl patch 后必须同步 helm values,避免 `helm upgrade` 覆盖修复。values 文件应作为唯一真相源。

5. **基准测试隔离**: 验证修复前应先隔离外部流量,避免并发请求干扰 TTFT 测试结果。
