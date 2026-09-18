# Prefill top-k v2 packed kernel 移植计划（PR #37889 → mi308x-1p1d-src）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port AMD's GLM DSA prefill top-k v2 packed kernel (PR #37889, `EricKing626/sglang:amd/topk-v2-prefill`) to our `mi308x-1p1d-src` branch. Upstream measured: prefill top-k launch −66%, e2e +4.9% throughput / −3.5% TPOT / −4.8% TTFT (ISL 70K, conc 4-64 geomean), GSM8k 0.937 vs 0.935. Our prefill trace: topk_transform = 10.4% of kernel time (5.48s/52.4s).

**Why not a straight cherry-pick:** 15 conflict hunks across 4 files (our W2 coop port + splitk diverged from upstream), AND two gfx942-specific blockers upstream did not hit (they tested gfx950/CDNA4):

1. **Cluster kernels**: `topk_v2.cuh` contains `topk_persistent_cluster_kernel` / `topk_small_batch_cluster_kernel` (`__cluster_dims__` + `cg::this_cluster()`). gfx942 is CDNA3 — **no thread-block clusters** — the JIT compile of the whole translation unit fails. Upstream tested on gfx950 (CDNA4, has clusters).
2. **Env gating**: the PR routes through `should_use_topk_v2()`, but our `server_args.py:5293` force-disables `SGLANG_OPT_USE_TOPK_V2` for all DSA on HIP (correctly — the paged/ragged v2 entries need clusters). Upstream's own final position (commit `6331e43081`) is also TOPK_V2=0 default on HIP. The packed entry is cluster-free and must be gated separately.

**Architecture:** semantic port of the packed kernel + a new env gate `SGLANG_DSA_USE_TOPK_V2_PACKED` (default off), cluster kernels guarded behind a cluster-support macro so the TU compiles on gfx942, dispatch routed only for packed PAGED extend on HIP. Same gray-release pipeline as CK GEMM: build `v0919-src-topk2p` → decode-1 PoC → canary → decode-0 → prefill-1 → prefill-0 (prefill is where the win is; decode unchanged).

**Tech Stack:** branch `mi308x-1p1d-src`, PR source `EricKing626/sglang@amd/topk-v2-prefill` (fetched as FETCH_HEAD), ti-builder, `sts_poc_mode.py`, `bench_poc_suite.py`, `bench_codex_200k_glm53.py`.

---

## 0. 前置事实（已核实）

| 项 | 状态 |
|---|---|
| QuickReduce 修复 | ✅ 已 cherry-pick（`4d9187adf4`，干净）——与本计划同一张镜像 |
| PR 源 | FETCH_HEAD = `a6d769e4ea`（5 个 commit：kernel 拆分 + USE_ROCM + 注释 + 测试标记 + chunked extend） |
| PR 核心改动 | `fb2eba8a14`：354+/168−，4 文件 |
| 我们 topk 现状 | decode = coop kernel（AOT，`fast_topk_v2`）；prefill = legacy `fast_topk_transform_fused`（hipify 直方图） |
| 冲突根因 | 我们有 W2 coop port + splitk；上游有 plan/cluster 重构 |
| gfx942 cluster | 无（CDNA3）。PR 在 gfx950（CDNA4）验证 |
| prefill topk 占比 | 10.4%（5.48s / 52.4s kernel time，9/8 trace，旧 kernel 数字） |

## 1. 硬约束

1. 源码进仓库，语义移植（不是 patch 脚本）。
2. 新 env `SGLANG_DSA_USE_TOPK_V2_PACKED`（默认 off）——不动 `SGLANG_OPT_USE_TOPK_V2`（保持 force-off 语义）。
3. cluster kernels 用宏门控（`#if !defined(USE_ROCM) || defined(SGL_HAS_CLUSTERS)` 或直接 `#ifndef USE_ROCM`——gfx942 上 plan 的 cluster 分支不可达，见 §2.3）。
4. decode 路径零改动（coop kernel 不动）。
5. 本波禁止 helm；灰度同 CK GEMM 流程。

## 2. Stage A — 源码（4 个文件）

### 2.1 `python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh`

- [ ] A1: 从 FETCH_HEAD 移植 `TopKPackedParams`、`topk_packed_kernel<kPDL>`、`TopKKernel::transform_packed` host 入口（参考 `git show fb2eba8a14` 的 + 侧）
- [ ] A2: cluster 门控：`topk_persistent_cluster_kernel` / `topk_small_batch_cluster_kernel` / `CLUSTER_TOPK_KERNEL` 宏 + `plan()` 的 cluster 分支包进 `#ifndef USE_ROCM`（gfx942 上 `topk_plan` 不会被调用——packed 路径不需要 plan；decode 走 coop kernel 不走这里）
- [ ] A3: `mask_head` / `head_residue`（packed 行前 16B 对齐残差掩码）随 params 一起移植

### 2.2 `python/sglang/kernels/ops/attention/dsv4/topk.py`

- [ ] A4: 注册 `topk_transform_packed` wrapper（`is_hip_runtime()` 时）+ `topk_transform_packed_v2(scores, seq_lens, page_tables, out_page_indices, page_size, *, row_starts, row_to_batch=None)` Python 入口

### 2.3 `python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py`

- [ ] A5: packed PAGED extend 路由（在现有 PAGED 分支后加）：
  - 条件：`_is_hip and envs.SGLANG_DSA_USE_TOPK_V2_PACKED.get() and topk_transform_method == PAGED and row_starts is not None and logits.stride(0) % 4 == 0 and 0 < topk <= 2048`
  - `row_to_batch`：`batch_idx_list is None → attn_metadata.token_to_batch_idx`；tensor → 自身；list → None（回 legacy）
  - **decode 条件保持逐字节不变**（`dsa_drop_wide_page_table` 依赖它）
- [ ] A6: `environ.py` 加 `SGLANG_DSA_USE_TOPK_V2_PACKED = EnvBool(False)`

### 2.4 测试

- [ ] A7: `test/registered/kernels/ops/attention/test_topk_v2.py` 移植 packed 用例（packed_rows / level_boundary / all_equal）
- [ ] A8: 新增 gfx942 编译冒烟：`_jit_topk_v2_module()` 在无 GPU 环境至少 import 不炸（cluster 门控生效的验证）

### 2.5 Dockerfile 断言 + commit

- [ ] A9: Layer 2 gate 加 `assert 'topk_packed_kernel' in r('kernels/jit/csrc/deepseek_v4/topk_v2.cuh'), 'topk2p-kernel';`
- [ ] A10: commit（含 QuickReduce `4d9187adf4`，一张镜像两个修复）

## 3. Stage B — 镜像 `v0919-src-topk2p`

- [ ] B1: ti-builder build（Layer 2b 重建 sgl-kernel：QuickReduce .cuh 修复 + topk_v2.cuh JIT 源）
- [ ] B2: push；不删旧 tag
- [ ] B3: 镜像内验证：`grep topk_packed_kernel` + `grep kQRFp16CastScaleLog2`（QuickReduce 标记）

## 4. Stage C — prefill-1 PoC（收益在 prefill，先 PoC prefill）

- [ ] C1: prefill-1 切 PoC 模式（`sts_poc_mode.py poc sglang-1p1d-prefill-1`）
- [ ] C2: 基线（env off）：`bench_codex_200k_glm53.py` t1 冷 ×3（TTFT 中位）+ needle
- [ ] C3: 开 env `SGLANG_DSA_USE_TOPK_V2_PACKED=1` 重建
- [ ] C4: 验证激活：日志/trace 确认 packed 路径命中（prefill topk kernel 名变化）
- [ ] C5: PoC ×3：**门禁 t1 冷 TTFT 中位 ≥ −2%（期望 −4~−5%）**，needle 不回退
- [ ] C6: 数值抽测：packed vs legacy topk 输出集合相等（torch.topk 参考）

## 5. Stage D — 灰度

- [ ] D1: prefill-1 恢复生产 + 观察 2h（冷 TTFT / warm HIT / 无错误）
- [ ] D2: prefill-0 同流程
- [ ] D3: decode 两台也带 env（decode 不走 packed 路径，env 无操作——保持镜像统一）
- [ ] D4: chart 固化 + commit

## 6. 回滚

| 级别 | 动作 |
|---|---|
| env 级 | `SGLANG_DSA_USE_TOPK_V2_PACKED=0` + 删 pod |
| 镜像级 | 回 `v0918-src-ckgemm` |

## 7. 门禁

| 门禁 | 阈值 |
|---|---|
| t1 冷 200K TTFT | 中位 ≥ −2%（期望 −4~−5%） |
| t2 warm HIT | 行为不变 |
| needle | 不回退 |
| packed vs legacy | 选点集合相等 |
| decode | 零变化（路径不经过） |
| 稳定性 | 2h 无错误 |

## 8. 风险

| 风险 | 缓解 |
|---|---|
| cluster 门控漏包 → gfx942 编译失败 | B3 镜像内编译冒烟 + A8 |
| packed 路径在 chunked extend（batch_idx_list）下选错行 | C6 集合相等测试 + needle |
| JIT 首次编译 stall prefill | PoC 里量 t1 首次 vs 后续（JIT cache 后应一致） |
| 与 W2 coop 冲突 | decode 路径零改动（A5 约束） |
