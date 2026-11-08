# ds41 mHC #1 launch fusion — image build assets (PURE PREPARATION)

**Nothing here is built, pushed, deployed or restarted in this round.** It is a
prepared asset for a future GO + frozen window. The live service
`ds41-flash.jmpti.woa.com` is m20 / 1P1D; a double-pod restart = 断服.

## What the image changes

Exactly two `.py` files, on top of the m20 image:

| file | change |
|---|---|
| `srt/models/deepseek_v4_mhc.py` | env-gated ON branch at the `mixes = GEMM*rsqrt` call site; **OFF arm byte-for-byte unchanged** |
| `kernels/ops/layernorm/mhc.py` | `+` fused TileLang kernel `hc_split_sinkhorn_fused_kernel`, wrapper `hc_split_sinkhorn_fused`, `ENV_MHC_FUSED_SINKHORN`, `fuse_enabled`, `import os`. **Shipped `hc_split_sinkhorn_kernel` / `hc_split_sinkhorn` / gfx1250 ports untouched.** |

No `.cu`, no AOT `.so`, no rebuild of `sgl_kernel`. `SGLANG_DS41_MHC_FUSED_SINKHORN`
defaults off (`!= "0"`), read per call, so the shipped image behaves exactly as
m20 until the env is set.

## Bit-exactness — ✅ GPU `torch.equal` MEASURED (2026-11-08)

The OFF chain computes `mixes = GEMM * rsqrt` as one fp32 mul (one launch/layer).
The ON chain passes the raw GEMM output and `rsqrt` to the fused kernel, which
lands `mixes_shared[j] = mixes_shared[j] * rs` into **fp32 shared memory** as a
standalone statement before the SAME `sigmoid`/comb/reduce (verbatim suffix).
Same single fp32 rounding → same `m` → identical `pre`/`post`/`comb`.

`verify/verify_mhc_fusion.py` GPU branch was run on a gfx942 MI308X offline
python: **3 seeds × 36 cases = 108 cases, `torch.equal` pre/post/comb whole-tensor
PASS**, `OFF=2 → ON=1` launch, `ON-arm dispatch` proven (no fallback false PASS).
TileLang's generated device source keeps the standalone fp32 landed mul and has
**0 `fma`/`fmuladd`** ⇒ the residual FMA/reorder doubt is resolved. HE+/GSM8K
are therefore waived (位精确类). See `manifests/ds41-mhc-fusion/gate-checklist.md`
§1/§2 for the full evidence and §7 for the reproducible command.

## ✅ m20 base bytes — READ, md5 PINNED (2026-11-08, not fabricated)

m20's two `.py` bytes were read from the pinned digest in ti-builder
`dind-builder4`
(`ds41-prod-1033-m20@sha256:5a0fcb8a2e12eafd6dab9f8ea632051a27152e63e342229fa506484c8d934374`)
and cross-checked against the **live m20 decode pod** and **fork HEAD
`d7dab9171c`** git blobs — all three agree:

```
mhc.py              = 04d8a21996483628daf38f1ff16d2f7d
deepseek_v4_mhc.py  = e71f21c856bce40895cdeb83f7a995e5
```

- `build.sh` still pulls those two files **from the pinned m20 DIGEST** into
  `m20src/` / `baseline/` at build time.
- `check_mhc_fusion.py` now **PINS these md5s** (`BASE_MD5`) and fails the build
  on a wrong digest / re-pushed tag — in addition to the dynamic OFF-arm /
  shipped-kernel delta check.
- `patch_mhc_fusion.py` asserts the pulled base is an **insertion-only ancestor**
  of each reviewed worktree file (difflib, no delete/replace) and then **emits the
  reviewed worktree bytes verbatim** — the exact bytes the GPU `torch.equal` was
  measured against. Earlier AST-lifting dropped the comment block and reordered
  a def; that byte drift has been fixed. Any base divergence fails loudly.

The full `patch → check → verify` chain was run against these REAL m20 bytes:
ALL PASS, staged `cmp` byte-identical to the worktree.

## Files

- `Dockerfile` — `ARG M20_BASE` (pin the m20 **digest**), COPY the 2 patched
  files + baseline, RUN `check_mhc_fusion.py` + `verify_mhc_fusion.py` (GPU-free).
- `build.sh` — **REFUSING** by default (`FUSION_BUILD_GO=1` + `M20_DIGEST`
  required); pull baseline, patch, check, verify, build on ti-builder dind pod,
  **NOT pushed**.
- `patch_mhc_fusion.py` — insertion-only base assertion + emits reviewed worktree
  bytes verbatim (single source of truth, byte-identical to GPU-verified code).
- `check_mhc_fusion.py` — **BASE_MD5 pin** + dynamic delta check (OFF arm verbatim,
  shipped kernels unchanged, env default-off, ON wiring, fallback guards).
- `verify_mhc_fusion.py` — GPU-free build-time AST + fp32 scalar proof.

## Verification already run

```
# 本机：对从 pinned m20 digest 拉出的 REAL base 字节（非 stand-in）
python3 docker/ds41-image/ds41-mhc-fusion/patch_mhc_fusion.py <real-m20-base> . <out>   # OK, cmp == worktree
python3 docker/ds41-image/ds41-mhc-fusion/check_mhc_fusion.py <patched-model> <base-model> <patched-mhc> <base-mhc>   # ALL PASS (md5 pin 命中)
python3 docker/ds41-image/ds41-mhc-fusion/verify_mhc_fusion.py <mhc.py> <deepseek_v4_mhc.py>   # ALL PASS
python3 verify/verify_mhc_fusion.py                              # local AST + 50054-pair scalar
bash manifests/ds41-mhc-fusion/verify-render.sh                 # ALL PASS (read-only)

# 集群 gfx942 离线 python（GPU 位精确，2026-11-08）
SGLANG_SRC=<staged-src> VERIFY_SEED=20261108 /opt/venv/bin/python3 verify/verify_mhc_fusion.py
#   GPU torch.equal: PASS (36 cases) × 3 种子 = 108 cases; OFF=2 ON=1 launch
```

`torch.equal` **已实测**（在集群），不再只是 by-construction + 静态举证。
build 容器内那条 GPU-free 脚本仍打印「本容器无 GPU」，因为构建阶段确实无 GPU —
它只是 build-time 静态门禁，位精确以集群 GPU 分支为准。

## Rollout (NOT this round)

**已完成**：GPU `torch.equal` PASS、m20 base 三源核实 + md5 pin、patch/check/verify
对真实 m20 字节跑通、verify-render ALL PASS。
**剩余仅 = GO + 冻结窗 + 同源同切原子重启**：`FUSION_BUILD_GO=1 M20_DIGEST=… bash
build.sh`（默认 REFUSING）→ 推送（独立 GO）→ `values.on.yaml` → 冻结窗内同源同切
双 pod 原子重启 → gate checklist §3~§6。回滚 SOP：env 设 `"0"` / 去掉 overlay —
**无需 rebuild**。三段可复制命令见 `manifests/ds41-mhc-fusion/gate-checklist.md` §7。
