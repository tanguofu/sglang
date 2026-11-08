# ds41 mHC #1 launch fusion — rollout manifests (PURE PREPARATION)

**Nothing here is applied.** No deploy, no restart, no rebuild, no 现网 change.
Live service `ds41-flash.jmpti.woa.com` is m20 / 1P1D. These files are staged
in this worktree; in a real rollout they would live in
`/Users/guofutan/ai-frameworks/dsv4-308x/manifests/ds41-mhc-fusion/`
(the chart is a different repo) and the image assets in that repo's
`docker/ds41-image/`.

## What / why

gfx942 (ts4, MI308X, ROCm, `torch.version.cuda is None`) 现网 mHC 走
`deepseek_v4_mhc.py:380-396` 的 else 分支（`SGLANG_DS41_MHC_FUSED_STATS="0"`、
`MHC_TRITON_MIX` 默认 0 ⇒ `use_fused_stats=False` ⇒ `:374` False）。`:396`
`mixes = (GEMM * rsqrt)` 是每层一个独立 fp32 elementwise mul launch，命中 decode
热点。

本融合（`kernels/ops/layernorm/mhc.py` 的 `hc_split_sinkhorn_fused_kernel` +
`hc_split_sinkhorn_fused`）把「未乘 rsqrt 的 GEMM 输出」和 `rsqrt`（`hc_rms_cast`
产 `[n,1]` fp32）分别传入，在 kernel 里 `mixes_shared[j] = mixes_shared[j] * rs`
**独立语句落地 fp32 shared memory**，再进入与原来**完全相同**的
`sigmoid`/comb/reduce（后缀 verbatim）。原链 `GEMM*rsqrt` 是一次 fp32 乘落地
fp32，融合后同一次 ⇒ 逐位相同 ⇒ 位精确 by construction。

`SGLANG_DS41_MHC_FUSED_SINKHORN` 默认关（`!= "0"`），per-call 读（gate
`_mhc_fused_sinkhorn_on_hip()`），OFF arm 逐字节复现现网链（`check_mhc_fusion.py`
坐实）。gfx1250(MI350X) 走 `_hc_split_sinkhorn_triton`（`:490`），ts4 不走、
保持原样；wrapper 内 `is_gfx1250_supported()` 守卫也会 fallback 到原链。

✅ **2026-11-08 GPU `torch.equal` 已在 gfx942 实测全量 PASS**：3 个独立种子
× 36 cases = 108 cases，pre/post/comb 全量逐 bit 相同；`OFF=2 → ON=1` launch；
`ON-arm dispatch` 证明 fused kernel 真跑（非 fallback 假 PASS）；TileLang 生成
设备源码 0 个 FMA、独立 fp32 落地乘保留。⇒ TileLang FMA/重排疑虑已消除、判为
**位精确类**、免 HE+/GSM8K。本机仍无 GPU（验证在集群**离线 python** 跑的，未进
serving 进程 / 不发 HTTP / 不发压）。详见 `gate-checklist.md` §1/§2/§7。

## ⚠️ 上一轮 no-op int64-widen 融合已清理（DEPRECATED）

`ds41-fusion-proto` worktree 里上一轮的 `SGLANG_DS41_FUSE_INDEX_K_LOC_WIDEN`
融合（`fp8_index_backend_hip.py` 的 int64-widen）**现网永不生效、收益精确 0**，
本轮已从 worktree **全部删除**，不要与真实现混淆：

- **原因**：`c2_out_loc` ← `_low_ratio_compression_metadata:447-449` 产
  `torch.where(..., raw_out_loc.to(int64)//ratio, -1)` = **int64**；
  `fused_pair_partners_and_slots:234` `slots=empty_like(out_loc)` = **int64**；
  而其 `loc_widen_fusable` guard 要求 **int32** 才融合 ⇒ 永远走 else 原链。
  且 `.to(int64)` 对 int64 tensor 本就不产生 launch ⇒ 无 launch 可省。
- **清理动作**：`git checkout` 还原 `fp8_index_backend_hip.py`；删除
  `index_k_fp8_loc_widen.py`、`verify/verify_loc_widen.py`、
  `docker/ds41-image/ds41-fusion-kernel/`、`manifests/ds41-fusion-kernel/`。
  本轮 `git status` 仅含下面「Files」的真实现。

## Files

| file | purpose |
|---|---|
| `values.base.yaml` | m20 user-values snapshot (verbatim from `dsv4-308x/manifests/ds41-c8-fixed-cost/values.m20.yaml`) |
| `values.on.yaml` | **one** env key under shared `env:` (`_helpers.tpl:66` merges into both pods = 同源同切) |
| `verify-render.sh` | read-only `helm template` A/B: 1 leaf → 2 env entries (1 per pod), all else identical, Helm 4.1.3 `N=0` trap absent |
| `gate-checklist.md` | 从严门禁（见下） |
| `README.md` | this file |

## Checks already run

```
# 本机（AST / scalar / render / 真实 m20 字节 build 链）
bash manifests/ds41-mhc-fusion/verify-render.sh                     # ALL PASS
python3 verify/verify_mhc_fusion.py                                  # local AST + 50054-pair scalar
python3 docker/ds41-image/ds41-mhc-fusion/patch_mhc_fusion.py ...   # OK, staged == worktree 逐字节
python3 docker/ds41-image/ds41-mhc-fusion/check_mhc_fusion.py ...   # ALL PASS (m20 base md5 PIN 命中)
python3 docker/ds41-image/ds41-mhc-fusion/verify_mhc_fusion.py ...  # ALL PASS

# 集群 gfx942 离线 python（GPU 位精确，2026-11-08）
SGLANG_SRC=/tmp/ds41verify/python VERIFY_SEED=20261108 \
  /opt/venv/bin/python3 verify/verify_mhc_fusion.py                 # GPU torch.equal: PASS (36 cases)
#   × 3 种子 = 108 cases; launches OFF=2 ON=1 saved=1; ON-arm fused kernel hits=1
```

m20 base 字节已从 **pinned digest**（ti-builder `dind-builder4`
`ds41-prod-1033-m20@sha256:5a0fcb8a…`）读取，并与 live m20 pod、fork HEAD
`d7dab9171c` 三源核实：`mhc.py=04d8a219…` / `deepseek_v4_mhc.py=e71f21c8…`；
`check_mhc_fusion.py` 已把它写成 **BASE_MD5 pin**（构建期强制，不是瞎编）。
完整构建链对真实 m20 字节跑通，staged 两文件与 GPU 验证用的 worktree `cmp` 逐字节相同。

`torch.equal` **已实测**（在集群），不再只是 by-construction + 静态举证。
三段可复制命令（GPU verify / ti-builder 构建 / 冻结窗同源同切原子重启）见
`gate-checklist.md` §7。

## Gate (summary — full version in gate-checklist.md)

1. ✅ GPU `verify/verify_mhc_fusion.py` → `torch.equal` pre/post/comb **PASS**
   (3 种子 108 cases，含 c=1 的 M=4×bs 阈值) + launch `OFF=2 → ON=1`。
2. ✅ 判为**位精确类**，免 HE+/GSM8K（GPU 全量 PASS 已达成）；若任一复跑失败
   立即退回非位精确类 → HumanEval+/GSM8K A→B→A ≥3 臂短窗交替。
3. [ ] conc=1/8/18/32 崩溃 0/N；**greedy + 默认采样双路径**逐字节文本一致
   (全 greedy 会短路 DSPARK verify 的 OOM 路径); hipFree 不动; 同源同切;
   冻结窗; A→B→A; 回滚 SOP = 去掉 env / 设 `"0"`，无需 rebuild。
4. ✅ TileLang 编译器行为已由 GPU `torch.equal` + 生成源码（0 FMA）双重坐实。

## Expected benefit (归档外推，非现网实测)

省每层 1 个 fp32 mul launch。**60/step 是 m8 bs=8 归档外推**、现网 c=1 未测、
约 **0.5% 量级**；禁用 2861/1905/135/331.8µs 旧数字。若 GPU `torch.equal`
失败 ⇒ 诚实收益 = 0，不上线。

## Rollout prerequisites (NOT this round)

**已完成**：GPU `torch.equal` PASS（§1）、m20 base 三源核实 + md5 pin、完整
build 链对真实 m20 字节跑通、verify-render ALL PASS、构建资产 finalize。
**剩余仅 = GO + 冻结窗 + 同源同切原子重启**：镜像构建 + 推送（`build.sh`，默认
REFUSING）、rebuilt image (2 changed .py files)、GO、冻结窗逐个确认、同源同切
double-pod restart（OnDelete STS + router 摘流）、freeze-window 切流、
`gate-checklist.md` §3~§6 全部条目。三段命令见 `gate-checklist.md` §7。
