# Gate checklist — mHC #1 launch fusion (mHC 数值高敏 / 位精确-by-construction)

**Class**: mHC 链喂给 sigmoid+sinkhorn，换实现历史上实测掉 3 分 HumanEval
（`deepseek_v4_mhc.py:56-60`：93.90→90.85）。本融合**由构造位精确**（同一次
fp32 乘、同序、强制 fp32 落地 shared）。

**2026-11-08 更新：GPU `torch.equal` 已在 gfx942 实测全量 PASS**（§1），
TileLang FMA/loop-fusion 重排疑虑已由实测 + 生成源码双重消除 ⇒ 判为
**位精确类**，HE+/GSM8K 打分门禁可免（§2）。本轮仍是**纯准备**：NO deploy /
NO restart / NO rebuild / NO 推送 / NO 现网变更。§0 未勾的只读前置、§3~§6
仍是上线前必做，不是「已完成」。

---

## 0. 前置（GO 之前，只读）

- [x] **m20 base 字节三源核实**（别瞎编 md5）：fork HEAD `d7dab9171c` git
  blob == live m20 decode pod `/sgl-workspace/sglang/python/sglang/...` ==
  ti-builder `dind-builder4`
  `ds41-prod-1033-m20@sha256:5a0fcb8a2e12eafd6dab9f8ea632051a27152e63e342229fa506484c8d934374`；
  `mhc.py=04d8a21996483628daf38f1ff16d2f7d`、
  `deepseek_v4_mhc.py=e71f21c856bce40895cdeb83f7a995e5`。
  `check_mhc_fusion.py` 已写成 **BASE_MD5 pin**（构建期强制，错 digest / 重推
  tag 直接 fail）。
- [x] **完整 build 链对真实 m20 字节跑通**：`patch_mhc_fusion.py`（插入-only
  断言 + 原样输出 reviewed 字节）→ `check_mhc_fusion.py`（md5 pin + OFF arm /
  shipped kernels delta）→ GPU-free `verify_mhc_fusion.py`，**ALL PASS**；
  staged 两文件与 GPU 验证用的 worktree `cmp` **逐字节相同**
  （`6de9053592f614651c5391d11e8ce33d` / `36783a43ff8069ab4086ea72e55456f1`）。
- [x] `bash manifests/ds41-mhc-fusion/verify-render.sh` → **ALL PASS**
  （read-only `helm template` A/B：共享 `env:` 双 pod 各 +1 条 env，其余字节不变）。
- [ ] **冻结窗**逐个确认（不能默认「团队已暂停」）；用服务端观测判空闲：
  `num_running_reqs` 连续两次 0 + TTFT 哨兵；不要用进程名
  （`feedback_no_process_name_mutual_exclusion.md` / 共享集群静默污染）。
- [ ] 确认现网 rev = m20、`restartCount=0`、两 pod 同源同切（同一镜像 digest）。

## 1. 第一门禁：GPU 位精确 —— ✅ 已实测 PASS（2026-11-08）

环境：`cls-bmmk3vtl-context` / `ds41-ts4-poc-decode-0`，MI308X **gfx942**，
`torch 2.9.1+rocm7.2.0` / `triton 3.7.0` / `tilelang 0.1.7.post3`，
`HIP_VISIBLE_DEVICES=0` 的**独立 python 进程**（不进 serving 进程、不发 HTTP、
不发压、tiny kernel）。`SGLANG_SRC=/tmp/ds41verify/python` = m20 源码基线 +
两个 worktree 文件（md5 见 §0，base `04d8a…`/`e71f2…`），逐字节核过。

- [x] `GPU torch.equal: PASS (36 cases, M in [1,4,8,16,18,32,64,128,256,512,2048,4096],
  pre/post/comb whole-tensor bit compare)`；**3 个独立种子**
  （`VERIFY_SEED=20261108 / 1 / 424242`）= **108 cases 全 PASS**。`M=4` 是 c=1
  的 M=4×bs 投机-verify 阈值（现网 DSPARK gamma=3 已开）。
- [x] `ON-arm dispatch proven: fused kernel hits=1, fallback hits=0,
  is_gfx1250_supported=False` ⇒ 排除「静默 fallback 造成假 PASS」。
- [x] `CUDA launches per call: OFF=2 ON=1 saved=1` ⇒ `:396` 的 mul launch 确实被折掉。
- [x] **TileLang 生成设备源码双重坐实**（cache `…/fc42ac84…/device_kernel.cu`）：
  line 20 `mixes_shared[tid] = mixes_shared[tid] * rs;` 独立落地 fp32 shared，
  line 23/27/29 下游 sigmoid/comb verbatim；`fma|fmuladd` 出现次数 off/on 均为
  **0** ⇒ 编译器未把乘收缩进 FMA、未 loop-fusion 重排。
- [x] eager microbench `OFF≈46us / ON≈38us` 仅作 launch 机制佐证，**不是现网收益**。
- [x] 位精确论证留档：AST 举证 + 50054 对 fp32 标量仿真 + 反事实（无落地会翻
  8361/50054 bit）在 `verify/verify_mhc_fusion.py` 本机臂。

复现命令见 §7 段 1。**TileLang 缓存 `OSError: Invalid cross-device link` 是
无害的 cache-save 警告**（`/root/.tilelang/cache/tmp` 与 `/root/.cache/sglang`
跨挂载 `os.replace` 失败），kernel 仍编译并执行，`torch.equal` 即证据。

## 2. 判据适用性 —— ✅ 判定「位精确类」，免 HE+/GSM8K

GPU `torch.equal` 对全部 shape/M 全量 PASS 且 launch 数确实掉 ⇒ 本次未换算术、
未换实现族，mHC 数值高敏的「换实现掉 3 分」理由不成立：

- [x] **免 HE+/GSM8K 打分门禁**；质量判据降级为 conc=1 greedy + 默认采样
  双路径逐字节文本一致（§3）。
- [ ] **若任一复跑失败 / 编译器行为存疑**：立即退回「非位精确类」，**必须**跑
  HumanEval+ / GSM8K A→B→A ≥3 臂短窗交替（双行文本一致不足以测 HE 掉分，
  见 `project_hardbench_gate_sensitivity.md`：HE+ 每 token 翻转率 0.122% vs
  GSM8K 1.553%）。A→B→A 同窗交替、无人抢流量、TTFT max 探测污染。

## 3. 稳定性 / 质量（从严）

- [ ] **崩溃 0/N**：四档并发 `c=1/8/18/32`，每档 N 次请求，
  `restartCount` 不变、无 5xx、无 503（decode `/health` 503 ≠ 进程死，按
  `reference_ds41_decode_health503.md` 判）、`dmesg` 无 oom-kill。
- [ ] **双路径**：每档同时跑 `temperature:0`（greedy）与**默认采样**
  （不传 temperature）。⚠️ 全 greedy 会短路 DSPARK verify 的 OOM 路径
  （`project_ds41_greedy_gate_blindspot.md`）——默认采样必须跑。
- [ ] **质量判据**：见第 2 节（位精确类免 HE/GSM8K）。默认采样比对固定 seed，
  greedy 严格逐字节。
- [ ] **hipFree 不动**：改动不新增 buffer / 不改 mem-fraction，
  `torch.cuda.mem_get_info` 的 hipFree 与 Δ 在 A/B 同窗应**逐字节相同**。
  ⚠️ 别照搬 dmabuf 类「掉 >8GiB 即停」判据（GDR 注册类），本类判据是
  「与基线偏差 ≈ 0」。
- [ ] **崩溃探针**：若出现段错误/乱码，区分 mHC/sinkhorn 与本次改动（本次只
  在 sinkhorn 输入加载前加一次 fp32 乘落地；OFF 链逐字节未动）。

## 4. 同源同切 / 切流

- [ ] prefill + decode **同一新镜像**（digest 相同），一次原子重启；
  1P1D 单 pod 重启 = 断服，须维护窗 + 切流（域名 rule → 备用/回滚 release）。
- [ ] `SGLANG_DS41_MHC_FUSED_SINKHORN` 放在**共享 `env:`**（`_helpers.tpl:66`
  merge 到双 pod），不是只给 decode；`verify-render.sh` 已证 render 恰好
  每 pod +1 条 env、其余字节不变（本机 ALL PASS）。
- [ ] STS 是 **OnDelete**：helm upgrade 改 env 后必须**手工删 pod**才生效，
  且 router AddWorker 超时后永不重试 → 先从 router 摘流再重启，600s bootstrap。
- [ ] 冻结窗内 A→B→A 至少各一轮（同窗、无人抢流量、TTFT max 探测污染）。

## 5. 回滚 SOP

1. `values.on.yaml` 去掉 / 把 `SGLANG_DS41_MHC_FUSED_SINKHORN` 设 `"0"`
   → `helm upgrade ... -f values.base.yaml`（**无需 rebuild 镜像**，env 默认关即 m20 行为）。
2. OnDelete：手工删两个 pod；从 router 摘流后再重启（AddWorker 超时即永久掉池）。
3. 校验：render 回到 `values.base.yaml` 的字节、四档 c=1/8/18/32 探活、
   默认采样 + greedy 双路径、`restartCount` 不变、hipFree 回到基线。
4. 若回滚后仍异常 → 回退镜像到 m20 digest（同源同切），再走一次第 1 节门禁。

## 6. STOP 条件（任一即停并如实报告）

- 复跑 GPU `torch.equal` 任一 shape/seed 失败，或 TileLang 编译器把乘并入
  FMA/重排（scalar 反事实已证无落地会翻 bit）；
- launch 数 `saved < 1`（目标不成立 / 打错点）；
- 双路径文本不一致、崩溃 ≥1 / 5xx / hipFree 偏差 ≠ 0；
- 冻结窗未确认、或有人在发压（共享集群静默污染）。

## 7. 三段可复制命令清单（纯准备 / 未执行，段 2、3 需 GO）

设 `W=/Users/guofutan/ai-frameworks/sglang/.claude/worktrees/ds41-fusion-proto`。

### 段 1 — GPU 跑 verify（✅ 已跑 PASS；可复现 / 上线前重跑）

```bash
W=/Users/guofutan/ai-frameworks/sglang/.claude/worktrees/ds41-fusion-proto
KC="kubectl --context cls-bmmk3vtl-context -n kube-system"
POD=ds41-ts4-poc-decode-0
# 1a) m20 源码基线复制到 pod /tmp（绝不改 /sgl-workspace），覆盖 2 个 worktree 文件 + verify
$KC exec $POD -- bash -c 'mkdir -p /tmp/ds41verify/python /tmp/ds41verify/verify; \
  cp -a /sgl-workspace/sglang/python/sglang /tmp/ds41verify/python/; \
  find /tmp/ds41verify/python/sglang -name "mhc.cpython-*.pyc" -delete; \
  find /tmp/ds41verify/python/sglang -name "deepseek_v4_mhc.cpython-*.pyc" -delete'
$KC exec -i $POD -- bash -c 'cat > /tmp/ds41verify/python/sglang/kernels/ops/layernorm/mhc.py'      < $W/python/sglang/kernels/ops/layernorm/mhc.py
$KC exec -i $POD -- bash -c 'cat > /tmp/ds41verify/python/sglang/srt/models/deepseek_v4_mhc.py'  < $W/python/sglang/srt/models/deepseek_v4_mhc.py
$KC exec -i $POD -- bash -c 'cat > /tmp/ds41verify/verify/verify_mhc_fusion.py'                   < $W/verify/verify_mhc_fusion.py
# 1b) 字节校验：应 = 6de9053592f614651c5391d11e8ce33d / 36783a43ff8069ab4086ea72e55456f1
$KC exec $POD -- md5sum /tmp/ds41verify/python/sglang/kernels/ops/layernorm/mhc.py /tmp/ds41verify/python/sglang/srt/models/deepseek_v4_mhc.py
# 1c) 单卡独立 python 跑 GPU 分支（不发 HTTP / 不发压）
$KC exec $POD -- bash -c 'cd /tmp/ds41verify; \
  HIP_VISIBLE_DEVICES=0 SGLANG_SRC=/tmp/ds41verify/python PYTHONPATH=/tmp/ds41verify/python \
  VERIFY_SEED=20261108 timeout 540 /opt/venv/bin/python3 -u /tmp/ds41verify/verify/verify_mhc_fusion.py'
# 期望：ON-arm dispatch proven / GPU torch.equal: PASS (36 cases) / launches OFF=2 ON=1 saved=1
# 换 VERIFY_SEED=1 / 424242 可复跑（已跑 3 种子全 PASS）
```

### 段 2 — ti-builder 构建（需 GO + M20_DIGEST；本轮未执行，build.sh 默认 REFUSING）

```bash
W=/Users/guofutan/ai-frameworks/sglang/.claude/worktrees/ds41-fusion-proto
M20_DIGEST=sha256:5a0fcb8a2e12eafd6dab9f8ea632051a27152e63e342229fa506484c8d934374
IMG=mirrors.tencent.com/ti-platform/sglang-glm52-308x:ds41-prod-1033-m20@$M20_DIGEST
# 2a) 本机纯校验：从 pinned digest 拉 m20 自己的两字节 → patch → check(md5 pin) → GPU-free verify
mkdir -p /tmp/ds41-mhc-fusion-base /tmp/ds41-mhc-fusion-staged
kubectl --context ti-builder -n default exec dind-builder4 -c builder -- sh -c "docker run --rm --entrypoint cat $IMG /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4_mhc.py" > /tmp/ds41-mhc-fusion-base/deepseek_v4_mhc.py
kubectl --context ti-builder -n default exec dind-builder4 -c builder -- sh -c "docker run --rm --entrypoint cat $IMG /sgl-workspace/sglang/python/sglang/kernels/ops/layernorm/mhc.py" > /tmp/ds41-mhc-fusion-base/mhc.py
md5sum /tmp/ds41-mhc-fusion-base/*.py   # = e71f21c8… / 04d8a219…
python3 $W/docker/ds41-image/ds41-mhc-fusion/patch_mhc_fusion.py /tmp/ds41-mhc-fusion-base $W /tmp/ds41-mhc-fusion-staged
python3 $W/docker/ds41-image/ds41-mhc-fusion/check_mhc_fusion.py \
  /tmp/ds41-mhc-fusion-staged/deepseek_v4_mhc.py /tmp/ds41-mhc-fusion-base/deepseek_v4_mhc.py \
  /tmp/ds41-mhc-fusion-staged/mhc.py /tmp/ds41-mhc-fusion-base/mhc.py          # ALL PASS
python3 $W/docker/ds41-image/ds41-mhc-fusion/verify_mhc_fusion.py \
  /tmp/ds41-mhc-fusion-staged/mhc.py /tmp/ds41-mhc-fusion-staged/deepseek_v4_mhc.py
# 2b) 真构建（GO 后、冻结窗内；产出镜像 NOT pushed，推送是另一个独立 GO）
cd $W/docker/ds41-image/ds41-mhc-fusion
FUSION_BUILD_GO=1 M20_DIGEST=$M20_DIGEST FUSION_IMAGE_TAG=ds41-prod-1033-mhc-fusion bash build.sh
# 产出：mirrors.tencent.com/ti-platform/sglang-glm52-308x:ds41-prod-1033-mhc-fusion
```

### 段 3 — 冻结窗同源同切原子重启（需 GO + 维护窗 + 切流；本轮未执行）

```bash
# 前置：段 1 PASS、段 2 镜像已构建并推送（GO）、冻结窗逐个确认、
#       两 pod 同源同切（同一新 digest）、先从 router 摘流。
KC="kubectl --context cls-bmmk3vtl-context -n kube-system"
REL=ds41-ts4-poc; NS=kube-system
CH=/Users/guofutan/ai-frameworks/dsv4-308x/charts/ds41-flash
BASE=$W/manifests/ds41-mhc-fusion/values.base.yaml   # 实际以集群 current user values 为准，先 diff
ON=$W/manifests/ds41-mhc-fusion/values.on.yaml
# 3a) 摘流：域名两 rule → 备用/回滚 release（08-decommission-plan.md §3.8）
# 3b) 只读 render 复核
bash $W/manifests/ds41-mhc-fusion/verify-render.sh                 # ALL PASS
# 3c) 同源同切：一次 helm upgrade 同时切新镜像 digest + 加共享 env（双 pod 各 +1 条）
helm -n $NS upgrade $REL $CH -f "$BASE" -f "$ON" --set image.tag=ds41-prod-1033-mhc-fusion
# 3d) OnDelete STS：改模板不自动重启，手工删两个 pod（原子同源同切）
$KC delete pod ds41-ts4-poc-decode-0 ds41-ts4-poc-prefill-0
# 3e) 等 600s bootstrap → 校验 render/env、四档 c=1/8/18/32 greedy+默认采样双路径、
#     restartCount 不变、hipFree 与基线逐字节相同
# 3f) 切流回域名，A→B→A 同窗；异常 → §5 回滚 SOP（去掉 env / 设 "0"，无需 rebuild）
```

## 预期收益（归档外推，非现网实测，勿夸大）

- 省 `:396` 每 mHC 层 1 个独立 fp32 elementwise mul launch（实测 `OFF=2 → ON=1`）。
- **60/step 是 m8 bs=8 归档外推**；现网 c=1 未测，约 **0.5% 量级**。
- **禁用** 2861/1905/135/331.8µs 等旧数字（不同改动 / 不同快照）。
- eager `46us→38us` 是共享 pod 的 microbench，**不能当现网步时收益**。
- 未部署 / 未重启 / 未改现网。
