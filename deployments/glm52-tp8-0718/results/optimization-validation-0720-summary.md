# GLM-5.2 2TP8 性能优化验证总结 — 2026-07-20

> 节点:node-21.234.170.19 + node-21.234.170.32(MI308X ×8,gfx942,`workload=glm52-prod`)
> 镜像:`mirrors.tencent.com/ti-platform/sglang-glm52-308x:fix-eagle-coredump-v3`
> sglang:`0.5.15.post1.dev20260718+gd7b9425529`,ROCm(HIP)7.2.0,host 7.2.4
> 方法:2 个 subagent 各占一台节点,各自做 before/after 对照,固定负载(28 并发,~12K ctx,max_tokens 256,streaming,temp 0.0,同 prompt/seed)

## 1. 验证维度与结论

| 维度 | 结论 | 收益 | 状态 |
|---|---|---|---|
| EAGLE+TP8 死锁复现 | **未复现**(28 并发 × 18min,1260 请求 0 失败,restartCount 0) | PR #31478 修复已在镜像内,稳定性 OK | ✅ |
| EAGLE 参数 A/B | **C4(steps=3→4)胜出** | **+8.6% 净 decode 吞吐**(569.5→618.5 tok/s) | ✅ 确定性收益 |
| hicache 写入策略 | **write_back→write_through_selective 激活 L2** | host_used 0→9280→12544,零吞吐/TTFT 回退 | ✅ 修复 dead-L2 bug |
| aiter gfx942 GEMM 调优 | not-found 缺口关闭,吞吐 +2.5%(噪声内) | 中性;方向修正:瓶颈不在 bf16 GEMM | ➖ 中性 |

## 2. EAGLE A/B 详细结果

固定负载:28 并发,~12K ctx,max_tokens 256,180s 窗口,warm cache。

| Config | steps | draft | topk | spec_accept_rate | spec_accept_length | gen_throughput | mean_wall | mean_TTFT | n_ok | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|
| C0(基线) | 3 | 4 | 1 | 0.588 | 2.765 | 569.5 tok/s | 21.9s | 1.399s | 224 | 当前生产配置 |
| C1 | 3 | 4 | 2 | — | — | — | — | — | — | ❌ 崩溃:DSA+page_size=64 与 topk>1 不兼容 |
| C2 | 3 | 6 | 1 | — | — | — | — | — | — | ⏭ 跳过:topk=1 时 draft 自动=steps+1=4,与 C0 同 |
| C3 | 3 | 8 | 1 | — | — | — | — | — | — | ⏭ 跳过:同 C2,自动塌缩 |
| **C4(胜出)** | 4 | 4 | 1 | 0.525 | 3.10 | **618.5 tok/s** | 20.05s | 1.434s | 252 | +8.6% |
| C5 | 4 | 8 | 2 | — | — | — | — | — | — | ❌ 崩溃:同 C1 |

### 关键发现:A/B 空间比预想窄

1. **topk=2 不可行**:GLM-5.2 用 DSA attention backend + page_size=64,`speculative_eagle_topk > 1 with page_size > 1` 只支持 `flashinfer/fa3/triton`。topk>1 需换 attention backend,超范围。
2. **draft=6/8 冗余**:sglang 在 topk=1 时强制 `speculative_num_draft_tokens = steps + 1`,所以 steps=3 时 draft 恒为 4,改 manifest 无效。C0/C2/C3 运行时是同一配置。
3. **唯一有效对比**:C0(steps=3)vs C4(steps=4),多一步 draft → accept_length 2.77→3.10 → 净吞吐 +8.6%。

## 3. hicache write_through_selective 验证

负载:同一 ~12K-token prompt 连发 5 次,cold start(每策略 fresh pod)。C4 EAGLE 参数恒定。

### write_back(当前生产,基线)

| rep | TTFT | wall | host_used | cache_hit_rate |
|---|---|---|---|---|
| 1 | 42.87s | 45.67s | 0.0 | 0.0 |
| 2 | 0.46s | 3.55s | 0.0 | 0.0 |
| 3 | 0.30s | 3.13s | 0.0 | 0.0 |
| 4 | 0.30s | 3.09s | 0.0 | 0.0 |
| 5 | 0.30s | 3.21s | 0.0 | 0.0 |

→ **确认 prod 的 dead-L2 bug**:host_used_tokens 全程 0,1.85M token 的 L2 容量完全浪费。write_back 只在 GPU(L1)evict 时写 host,而 GPU 用量 ~1% 从不 evict → L2 形同虚设。L1(GPU/radix)cache 正常(TTFT 42.87s cold → 0.30s warm)。

### write_through_selective(推荐)

| rep | TTFT | wall | host_used | cache_hit_rate |
|---|---|---|---|---|
| 1 | 41.74s | 44.74s | 0.0 | 0.0 |
| 2 | 0.46s | 3.33s | **9280.0** | 0.0 |
| 3 | 0.30s | 3.24s | 9280.0 | 0.0 |
| 4 | 0.30s | 3.32s | 9280.0 | 0.0 |
| 5 | 0.30s | 3.16s | 9280.0 | 0.0 |

→ **L1→L2 写入路径激活**:host_used_tokens 0→9280(rep 2),~12K prompt 的可复用前缀选择性写入 host。180s 28 并发负载下增长到 12544。**bug 修复**。

### 回退检查(C4 参数,180s 28 并发)

| 策略 | gen_throughput | mean_wall | mean_TTFT | host_used | restartCount |
|---|---|---|---|---|---|
| write_back | 618.5 tok/s | 20.05s | 1.434s | 0 | 0 |
| write_through_selective | 617.1 tok/s | 20.37s | 1.438s | 12544 | 0 |

→ 吞吐 −0.2%(噪声),TTFT +0.004s(噪声),**零回退**,无崩溃。

### 诚实 caveat

- `cache_hit_rate` gauge 全程读 0,不可靠 —— TTFT 才是真信号(42s cold → 0.30s warm 证明 L1 cache 工作)。
- L2 的 read-back 收益在本次低负载单前缀测试中没体现(L1 没 evict 压力,prefix 留在 L1,L2 不被读回)。L2 真正价值要在 L1 evict 压力下(更重/更多样前缀负载)才显现。但**激活写入路径是必要修复,且零成本**。

## 4. aiter gfx942 GEMM 调优

### 问题

`/tmp/aiter_configs/bf16_tuned_gemm.csv` **全是 gfx950 行,gfx942(MI308X)零调优配置**。decode 每个 bf16 GEMM 走默认 torch kernel fallback。日志实锤:`[aiter] shape M:30/112/120, N:256, K:6144 ... not found tuned config ... using torch solution:0`。

### 调优

- 入口:`/sgl-workspace/aiter/csrc/gemm_a16w16/gemm_a16w16_tune.py`(`aiter/tuned_gemm.py` 只做 lookup,非调优)
- 方法:padded-M bucket(pad0 向上取整到 16 的倍数),调 pad0 桶 {16,32,48,64,80,96,112,128}
- 28 个 gfx942 K=6144 shape 调优,~90s,0 失败
- 重载机制:env `AITER_CONFIG_GEMM_BF16=/data/aiter_configs/bf16_tuned_gemm.csv` 绕过每次启动重新生成 /tmp 的逻辑

### 结果

| | BEFORE(untuned) | AFTER(tuned) |
|---|---|---|
| 稳态吞吐 | 99.8 tps | 102.3 tps(+2.5%,噪声内) |
| N=256/K=6144 not-found | 大量 M | **0**(缺口关闭) |
| spec_accept_rate/length | — | 0.629 / 2.887(健康) |

### 方向修正(重要)

**吞吐中性,因为 decode 瓶颈不在 bf16 GEMM**:
1. untuned 时已 fallback 到 torch 默认 kernel,速度与调优 kernel 相当。
2. decode bf16 GEMM 占 decode 时间比例小,**attention / allreduce / MoE dispatch 才是主导**。

→ 下一阶段深挖方向:attention(DSA backend,FP8 KV cache decode)/ allreduce(TP8 通信)/ MoE dispatch。

## 5. 生产配置建议(19+32 生产对)

```diff
  # EAGLE: C0 → C4,+8.6% decode 吞吐
- --speculative-num-steps 3 --speculative-num-draft-tokens 4
+ --speculative-num-steps 4 --speculative-num-draft-tokens 4
  --speculative-eagle-topk 1                                    # 保持,topk=2 不可行(DSA)

  # hicache: 激活 L2,零回退
- --hicache-write-policy write_back
+ --hicache-write-policy write_through_selective

  # 可选(中性,消除 not-found 缺口,代码卫生)
  # env AITER_CONFIG_GEMM_BF16=/data/aiter_configs/bf16_tuned_gemm.csv
```

## 6. 产物清单

| 产物 | 路径 |
|---|---|
| C4 + selective 最终 manifest | `/tmp/test32-worker.yaml`(opt-32) |
| aiter tuned csv(2381 行) | `/data/aiter_configs/bf16_tuned_gemm.csv`(node 19/32 hostPath) |
| aiter 调优结果明细 | `results/tune19-aiter-gemm-bf16-gfx942-results.json` |
| node19 验证 runbook | `configs/pd-manifests/node19-verify-runbook.md` |
| tune19 manifest(含 env override) | `configs/pd-manifests/tune19-sglang-0.yaml` |
| 本总结 | `results/optimization-validation-0720-summary.md` |

## 7. 下一阶段(待验证)

aiter 调优中性证明 decode 真正瓶颈在 attention/allreduce/MoE dispatch。下一阶段在 19/32 上深挖:
- **attention**:DSA backend,FP8 KV cache decode 路径(aiter attention kernel)
- **allreduce**:TP8 通信大头,`--enable-aiter-allreduce-fusion` 已开但仍有空间
- **MoE dispatch**:GLM-5.2 是 MoE,expert 路由开销

这与 ANALYSIS.md 结论(reasoning token 占 98%)一致 —— decode 慢主要是 token 量 + attention 计算。
