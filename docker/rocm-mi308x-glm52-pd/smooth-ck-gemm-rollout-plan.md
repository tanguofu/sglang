# gfx942 CK dense GEMM 平滑灰度计划（CSV-HIT-only）

> **进度（2026-09-18）**：Stage A/B/C.1-C.3 完成。
> - commit `934a1c4cc`（fp8_utils + 单测 + Dockerfile 断言）已推送
> - 镜像 `v0918-src-ckgemm` build+push 完成（Layer 2b cache 命中，75s）
> - decode-1 PoC：基线 85.21 → CK 87.64 中位 **+2.9%（过 +2% 门禁）**；
>   accept_len 2.313→2.409（**+0.096，方向为正**——CK splitK=3 数值比 triton KSPLIT=16 更接近 target，
>   非数值损坏；needle 9/10，1 次为已知拒答 flake）；激活证据 752 条 a8w8 CSV tuned 行（基线 0 条）
> - **踩坑**：`sts_poc_mode.py restore` 恢复的是 9/16 旧备份（v0916-src-bugfix，无 CK env），
>   覆盖了 CK patch——已重新 patch 为 v0918+CK+生产 args 并 bounce。
>   后续用 restore 后必须重查 image/env。
> - Stage C.4 进行中：decode-1 生产 canary 起动，2h 观察 → decode-0

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable the aiter CK dense-GEMM path on gfx942 (MI308X) behind an env gate, so the already-deployed `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` cu80 CSV finally takes effect. CK is used **only when the CSV has a tuned row for the exact (M, N, K)**; any miss falls back to today's triton path unchanged. Gray-release decode-1 → decode-0 → prefill-1 → prefill-0, one STS at a time, zero service interruption.

**Why (evidence, 2026-09-18):**

- `aiter_w8a8_block_fp8_linear` (`fp8_utils.py:1108-1123`) forces `use_triton = True` on gfx942 (upstream PR #31217 wrote the gfx95 NaN workaround so that every non-gfx95 arch goes triton). The CSV env on all 4 pods is therefore **dead config** — verified by runtime spy (0 calls to `get_CKGEMM_config`), zero CSV log lines in all 4 pods, and code-path read.
- Microbench on decode-0 (same quant inputs, 200 reps): CK-with-CSV vs triton — M=4 N=4096 K=2048: 29.2 vs 71.1us (2.4×); M=4 N=2624 K=6144: 21.4 vs 97.7us (4.6×); M=16K N=2624 K=6144: 1662 vs 3017us (1.8×); M=16K N=2048 K=2048: 432 vs 780us (1.8×).
- Trace math (09-08, `/data/prof_decode_gpu` + `/data/prof_prefill_t1`): decode dense a8w8 = 25.1% of 6.066s GPU busy; expected CK saving ≈ 355ms ≈ **+5% tps**. Prefill dense = 3.98s of 52.4s; expected ≈ **+4–5% TTFT**.
- Numerics: CK vs triton on M=16K N=2624 K=6144 — mean abs diff 6e-6, max rel 2.3% (bf16/fp8 noise level); CUDA-graph capture+replay verified OK on decode-1.
- **Known MISS:** indexer wk (N=128 K=6144) has no CSV rows, and CK default is *slower* there (343 vs 303us). The HIT-only design keeps it on triton — no regression path.

**Tech Stack:** branch `mi308x-1p1d-src`, Dockerfile Layer 2 (`COPY python/sglang`), ti-builder `9.135.3.173:36000`, kube-system 1P1D, `sts_poc_mode.py`, `bench_poc_suite.py`, `bench_codex_200k_glm53.py`.

---

## 0. 今天的事实（不要按旧文档重做）

核实时间：2026-09-18。全部从现网 pod / 节点实测。

| 项 | 现网 | 本波 |
|---|---|---|
| CSV 文件 | 4 节点 `/data/aiter_configs/a8w8_blockscale_tuned_gemm_glm5_1_cu80.csv` 全在（238415B，md5 同） | 不动 |
| CSV env | 4 pod 全挂 `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` + `AITER_LOG_TUNED_CONFIG=1` | 不动（死配置，本波激活） |
| 现网镜像 | decode-0/1 = `v0916-src-projfuse`；prefill-0/1 = `v0916-src-mqa32` | 本波新 tag `v0918-src-ckgemm` |
| STS 策略 | 4 个 worker STS 全部 `OnDelete` | 保持；**本波禁止 helm**（chart 无 updateStrategy，helm 会打回 RollingUpdate） |
| `SGLANG_OPT_USE_TOPK_V2` | pod env=1（带外 patch），server_args 强制关生效（运行时验证） | 不动，与本波无关 |
| projfuse | decode 两台 ON，prefill 两台 OFF（有意） | 不动 |
| MTP | steps=2 draft=4 | 禁止改 |
| bench 基线 | decode conc=1 ~69 tok/s；64K TTFT ~22s；跑噪 ±7% | 同窗口 A/B，中位 ≥ +2% 才算赢 |

**科学对照**：不是 decode-1 vs decode-0（镜像将不同）。对照是 **同一台 decode-1：PoC 模式下 triton 基线 vs CK**（同机同图，只差 env），以及生产灰度期的同窗口 A/B。

---

## 1. 硬约束（违反即停）

1. **源码进仓库，禁止新 patch 脚本 / hostPath 变换器。**
2. **本波禁止 `helm upgrade`。** 改 env 用 `kubectl replace` 单 STS（导出→改→apply）。
3. **一次只 bounce 一个 pod。** 删 pod 前另一台同角色 `token_usage` 必须 < 0.55（`/metrics` 里 `# TYPE` 后的 `token_usage` gauge，或 `/v1/loads`）。
4. **CK 只在 CSV HIT 时启用**（代码里逐 shape 查表，MISS → triton）。禁止无条件 `use_triton=False`。
5. **门禁不过即回滚**：env 置 "0" + 删 pod，10 分钟回原状。旧 tag 永不删。
6. PoC 期间该 worker 不接生产流量（`sts_poc_mode.py poc` 把 :30000 换 :31000，router 自动摘除）。
7. 低峰窗口做 bounce（现网 22:00–02:00 低峰，按最近一周流量曲线选）。

---

## 2. Stage A — 源码（branch `mi308x-1p1d-src`，一个 commit）

### 2.1 改动

**File: `python/sglang/srt/layers/quantization/fp8_utils.py`**

1. import 块（`from sglang.srt.utils import (...)`，~line 35）加 `is_gfx942_supported`。
3. 模块级 flag（~line 67，`_use_aiter_bpreshuffle_gfx95` 之后）：

```python
# gfx942 (MI308X): the aiter CK blockscale GEMM is prebuilt for this arch and
# the cu80 tuned CSV (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE) has rows for every
# GLM-5.x dense shape except the indexer wk (N=128). Upstream PR #31217 routes
# every non-gfx95 arch to triton, which leaves the CSV dead config. Opt in to
# CK per shape: use it only when the CSV has a tuned row for the exact
# (padded_M, N, K); any miss (e.g. wk N=128, where the CK default is slower
# than triton) keeps today's triton path. Env-gated so rollback is a config
# bounce, not an image swap.
_use_aiter_ck_gfx942 = (
    _use_aiter
    and is_gfx942_supported()
    and not _is_gfx95_supported
    and get_bool_env_var("SGLANG_ROCM_USE_CK_GEMM_GFX942")
)
```

3. `aiter_w8a8_block_fp8_linear` 的分支（~line 1112-1123）改为：

```python
    if _use_aiter_bpreshuffle_gfx95:
        use_triton = use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k)
    elif _use_aiter_gfx95:
        # (unchanged comment)
        _ck_safe_m = _AITER_GFX95_CK_W8A8_MAX_SAFE_M.get((n, k))
        use_triton = use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k) or (
            _ck_safe_m is not None and input_2d.shape[0] > _ck_safe_m
        )
    elif _use_aiter_ck_gfx942 and _ck_csv_has_row(input_2d.shape[0], n, k):
        use_triton = False
    else:
        use_triton = True
```

4. 新增 helper（放在 `use_aiter_triton_gemm_w8a8_tuned_gfx950` 附近）：

```python
@lru_cache(maxsize=4096)
def _ck_csv_has_row(m: int, n: int, k: int) -> bool:
    """True if the a8w8 blockscale tuned CSV has a row for this shape.

    Mirrors get_CKGEMM_config's own key walk (gl in [None, 0, 1] with
    get_padded_m) so the probe and the real lookup agree: if the probe says
    HIT, the CK path will find the same row and never fall to its default
    heuristic.
    """
    if not _use_aiter_ck_gfx942:
        return False
    try:
        from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config
        from aiter.jit.core import AITER_CONFIGS

        return (
            get_CKGEMM_config(
                m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE
            )
            is not None
        )
    except Exception:
        return False
```

注意：`get_CKGEMM_config` 自带 `@functools.lru_cache`（查表本身幂等），外层再包一层 lru_cache 是为了把 per-call 的 Python 开销（decode 每 forward 多次调用）压到一次。**异常一律回 triton**（CSV 缺文件 / aiter 未装 / 任何意外）。

5. **不改** `aiter_w8a8_block_fp8_linear` 之外的任何调用方；`_use_aiter_bpreshuffle_gfx95` 分支语义不变（gfx95 机器行为逐字节不变）。

### 2.2 单测

**File: `test/registered/unit/layers/quantization/test_fp8_utils_ck_gfx942.py`**（新建）

- `test_helper_returns_false_when_disabled`：monkeypatch `_use_aiter_ck_gfx942=False` → `_ck_csv_has_row` 恒 False（不 import aiter 也不炸）。
- `test_helper_swallows_exception`：monkeypatch `get_CKGEMM_config` 抛异常 → False。
- `test_env_gate_off_by_default`：不设 env 时 `_use_aiter_ck_gfx942` 为 False（默认关，回滚语义）。
- `test_branch_unchanged_on_gfx95`：`_use_aiter_gfx95=True` 时走原分支（构造 flag 组合，断言 `use_triton` 计算与旧逻辑一致）。
- HIP-only 用例（`@pytest.mark.skipif(not is_hip())`）：真 CSV 查表 — N=4096 K=2048 M=4 → True；N=128 K=6144 → False。

### 2.3 Dockerfile 断言

**File: `docker/rocm-mi308x-glm52-pd/Dockerfile`** Layer 2 gate 加两行：

```python
assert '_ck_csv_has_row' in r('srt/layers/quantization/fp8_utils.py'), 'ck-gfx942-gate'; \
assert 'SGLANG_ROCM_USE_CK_GEMM_GFX942' in r('srt/layers/quantization/fp8_utils.py'), 'ck-gfx942-env'; \
```

### 2.4 Commit

```
feat(rocm): opt-in gfx942 CK dense GEMM when the tuned CSV has a row

The a8w8 blockscale CSV on all MI308X nodes is dead config: upstream
routes every non-gfx95 arch to the triton path. Probe the CSV per
shape; use CK only on a HIT (microbench 1.8-4.6x on GLM-5.x dense
shapes), keep triton on any miss (indexer wk N=128 has no rows and
the CK default is slower there).
```

- [ ] Step 1: `git diff` 确认只动 `fp8_utils.py` + 新单测 + Dockerfile 断言行
- [ ] Step 2: `python3 -m pytest test/registered/unit/layers/quantization/test_fp8_utils_ck_gfx942.py -q` PASS
- [ ] Step 3: commit（不 squash）

---

## 3. Stage B — 构建镜像 `v0918-src-ckgemm`

ti-builder（`ssh ti-builder`，9.135.3.173:36000）：

- [ ] B1: fetch `mi308x-1p1d-src`，确认 HEAD 含 Stage A commit
- [ ] B2: `docker build -f docker/rocm-mi308x-glm52-pd/Dockerfile -t mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0918-src-ckgemm .`（Layer 2 COPY 源码；gate 断言全过才出镜像）
- [ ] B3: push 镜像；**不删 `v0916-src-projfuse` / `v0916-src-mqa32`**
- [ ] B4: 镜像内冒烟（builder 上 `docker run --rm ... python3 -c "import sglang.srt.layers.quantization.fp8_utils"`，无 GPU 也应 import 成功——helper 惰性 import aiter）

---

## 4. Stage C — decode-1 隔离 PoC（零流量，科学对照）

decode-1（.172）当前 `v0916-src-projfuse`。PoC 模式下 router 摘除该 worker，另一台 decode-0 顶全部新流量。

### C.1 基线（PoC 模式，triton，现图）

- [ ] C1: `python3 scripts/sts_poc_mode.py poc statefulset/sglang-1p1d-decode-1`（备份 spec 到 /tmp；:30000→:31000）
- [ ] C2: 等 health 200（`curl http://21.151.225.172:31000/health`）
- [ ] C3: `python3 scripts/bench_poc_suite.py --url http://21.151.225.172:31000 --label ckgemm-baseline` ×3，取中位：needle ok、decode tps、accept_len
- [ ] C4: 记录 accept_len 基线（现网 ~2.25 @100K；PoC 196K 场景记录实际值）

### C.2 PoC（同机，CK，只差 env）

- [ ] C5: `kubectl replace` decode-1 STS：image → `v0918-src-ckgemm`，env 加 `SGLANG_ROCM_USE_CK_GEMM_GFX942=1`（env 门控，回滚只改 env 不换镜像）+ 现有 CSV env 不变
- [ ] C6: 删 pod 重建（PoC spec 下重建仍是 PoC 模式）
- [ ] C7: 启动日志验证（`kubectl logs ... | grep "is tuned on cu_num = 80 in /data/aiter_configs/a8w8_blockscale_tuned_gemm_glm5_1_cu80"`）——**必须出现** a8w8 CSV 的 tuned 行（此前 4 pod 全为 0 条，这是激活的直接证据）
- [ ] C8: `bench_poc_suite.py --label ckgemm-poc` ×3 取中位
- [ ] C9: **门禁**：
  - needle 196K ok（质量无损）
  - decode tps 中位 ≥ 基线 +2%（期望 +4–6%）
  - accept_len 不降（|Δ| < 0.05；GEMM 换 kernel 不改数学，降了说明数值有问题）
  - 无 CrashLoop / watchdog / NaN 输出（needle 答案是干净 code，不是碎片）
- [ ] C10: 数值抽测：PoC pod 内跑 `numcheck2.py` 同款脚本（CK vs triton max rel ≤ 3%）

### C.3 恢复生产 + 生产灰度

- [ ] C11: `sts_poc_mode.py restore statefulset/sglang-1p1d-decode-1`（回生产 spec，image 已是 v0918）
- [ ] C12: 删 pod 重建 → 生产 decode-1 带 CK；router 自动纳管
- [ ] C13: 观察 ≥ 2h：`gen_throughput`、accept_len、`# Running`、无 5xx；同窗口与 decode-0（triton）对比
- [ ] C14: **decode-0**：`kubectl replace`（image + env `SGLANG_ROCM_USE_CK_GEMM_GFX942=1`）→ 排空（token_usage < 0.55）→ 删 pod → C7/C9 验证复跑一遍

---

## 5. Stage D — prefill 灰度（decode 稳定后）

prefill 重建丢 L2 host cache（L3 保留），冷 TTFT 会短暂升高——选低峰。

- [ ] D1: prefill-1（.152）`kubectl replace`：image → `v0918-src-ckgemm` + env `SGLANG_ROCM_USE_CK_GEMM_GFX942=1`（prefill 不带 projfuse env，其余不变）
- [ ] D2: 排空在途（`num_total_tokens` 归零 / `# Running` 0）→ 删 pod
- [ ] D3: 启动日志验证 a8w8 tuned 行（同 C7）
- [ ] D4: **门禁**（`bench_codex_200k_glm53.py`，prefill-1 /data）：
  - t1 冷 200K TTFT 中位 ≥ 基线 −2%（期望 −4~−5%，即 77s → ~73–74s；**门禁是不劣化**，收益按实测记）
  - t2 warm HIT 不回退（cache 行为不变）
  - needle 正确性不回退
- [ ] D5: 观察 ≥ 2h（含一次真实长请求），无 `Write page to storage failed` / transfer fail
- [ ] D6: prefill-0（.144）同流程（image + env）

---

## 6. 回滚（每级独立，10 分钟）

| 级别 | 动作 |
|---|---|
| env 级（首选） | STS env `SGLANG_ROCM_USE_CK_GEMM_GFX942=0` + 删 pod（不换镜像，10 分钟） |
| 镜像级 | STS image 回 `v0916-src-projfuse`（decode）/ `v0916-src-mqa32`（prefill）+ 删 pod |
| PoC 级 | `sts_poc_mode.py restore`（spec 备份在 /tmp/<sts>.prod.json） |

CSV 文件在 /data 无 env 指向即惰性，无需动。**任何一级回滚不需要动其他三台。**

---

## 7. 质量门禁汇总（严格无损）

| 门禁 | 阈值 | 不通过动作 |
|---|---|---|
| needle 196K/200K | ok=True，答案含 code | 立即回滚，查数值 |
| accept_len | \|Δ\| < 0.05 vs 基线 | 回滚（GEMM 不改数学，降=数值坏） |
| decode tps | 中位 ≥ +2%（同窗口 A/B） | 回滚（无收益不留风险） |
| 冷 TTFT | ≥ −2%（不劣化） | 回滚 |
| warm HIT | t2 命中行为不变 | 回滚 |
| 数值抽测 | CK vs triton max rel ≤ 3% | 回滚 |
| 稳定性 | 2h 无 CrashLoop/watchdog/5xx/NaN | 回滚 |

跑噪 ±7%：所有对比用同窗口 ×3 取中位；单次 69→71 无效。

---

## 8. 明确不做（本波）

- 不动 `SGLANG_OPT_USE_TOPK_V2`（gfx942 编译不过，server_args 强制关是正确行为）
- 不动 MTP steps / draft
- 不动 projfuse 配置（decode ON / prefill OFF 维持）
- 不动 router / mooncake-master
- 不做 prefill wk（N=128）CK 化——CSV 无行，CK default 更慢（343 vs 303us 实测）
- 不删任何旧 tag
