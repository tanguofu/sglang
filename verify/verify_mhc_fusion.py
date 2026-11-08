#!/usr/bin/env python3
"""A/B verify for the mHC #1 launch fusion (fold ``GEMM * rsqrt`` into sinkhorn).

Two modes, chosen automatically:

* GPU cluster (torch + triton + tilelang + a HIP device): runs the real
  ``hc_split_sinkhorn`` (OFF) and ``hc_split_sinkhorn_fused`` (ON) kernels A/B
  and asserts ``torch.equal`` on WHOLE ``pre`` / ``post`` / ``comb`` tensors for
  many token counts / M (including c=1's M=4*bs speculative-verify threshold),
  then measures CUDA launch count and CUDA-event time.  This is the ONLY mode
  that can claim bit-exactness has actually been MEASURED.
* No GPU / no triton (this laptop, hip=None): AST exhibit that the fused
  TileLang kernel is the shipped kernel + exactly one fp32-landing block, that
  the entire sigmoid/comb/reduce/sinkhorn suffix is byte-identical, and that the
  OFF arm in deepseek_v4_mhc.py is the untouched chain; plus an fp32 scalar
  element-by-element simulation of the landed multiply.  It prints
  ``未在 GPU 实测位精确`` and NEVER claims a torch.equal was run.

Usage (cluster):  SGLANG_SRC=/sgl-workspace/sglang/python python3 verify/verify_mhc_fusion.py
Usage (laptop):   python3 verify/verify_mhc_fusion.py          # auto local
Override source:  SGLANG_SRC=<python dir containing sglang/>

IMPORTANT: TileLang could in principle still fuse the standalone ``raw*rs``
loop into a downstream FMA or reassociate it (loop fusion / fast-math).  The
shared-memory store is the intended fp32 landing but its behaviour cannot be
proven 100% statically -- the GPU ``torch.equal`` branch below is mandatory
before any rollout.  See manifests/ds41-mhc-fusion/gate-checklist.md.

MEASURED RESULT (2026-11-08): GPU branch RUN and PASS on gfx942 MI308X
(``ds41-ts4-poc-decode-0``, offline python, ``HIP_VISIBLE_DEVICES=0``, no HTTP,
no load) with torch 2.9.1+rocm7.2.0 / triton 3.7.0 / tilelang 0.1.7.post3:
  * ``GPU torch.equal: PASS (36 cases, M in [1..4096], pre/post/comb
    whole-tensor)`` x 3 seeds (20261108 / 1 / 424242) = 108 cases,
  * ``ON-arm dispatch proven: fused kernel hits=1, fallback hits=0``
    (rules out a silent-fallback false PASS),
  * ``CUDA launches per call: OFF=2 ON=1 saved=1``,
  * TileLang generated device source keeps the standalone fp32 landed mul and
    has 0 ``fma``/``fmuladd`` -> no FMA contraction / loop-fusion reorder.
Evidence + reproducible commands: gate-checklist.md §1/§7.  VERIFY_SEED env
var re-runs the GPU branch with independent data.
"""

from __future__ import annotations

import ast
import math
import os
import random
import statistics
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.dirname(HERE)
DEFAULT_SRC = os.path.join(WORKTREE, "python")
SRC = os.environ.get("SGLANG_SRC", DEFAULT_SRC)
MHC_PATH = os.path.join(SRC, "sglang", "kernels", "ops", "layernorm", "mhc.py")
MODEL_PATH = os.path.join(SRC, "sglang", "srt", "models", "deepseek_v4_mhc.py")

ENV = "SGLANG_DS41_MHC_FUSED_SINKHORN"
OFF_LINE = "mixes = (row_uniform_mhc_mix(x_flat, hc.fn) * rsqrt).unsqueeze(1)"


def f32(x: float) -> float:
    """Round a Python (f64) float to the nearest IEEE fp32, exactly once.

    A finite product that exceeds the fp32 range becomes +/-inf, exactly as a
    real fp32 multiply would (struct raises OverflowError for that case).
    """
    x = float(x)
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def sigmoid(arg: float) -> float:
    """Numerically stable sigmoid with an fp32-rounded result (no exp overflow)."""
    if arg >= 0:
        return f32(1.0 / (1.0 + math.exp(-arg)))
    z = math.exp(arg)
    return f32(z / (1.0 + z))


# ---------------------------------------------------------------------------
# AST exhibit (GPU-free, import-free)
# ---------------------------------------------------------------------------


def _prim_with_body(tree, outer_name):
    """Return (args, with-body stmts) of the inner @T.prim_func of a kernel."""
    outer = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == outer_name
    )
    inner = next(n for n in outer.body if isinstance(n, ast.FunctionDef))
    assert isinstance(inner.body[0], ast.With), outer_name
    return [a.arg for a in inner.args.args], inner.body[0].body


def _unparse(stmts):
    return ast.unparse(ast.Module(body=list(stmts), type_ignores=[]))


def _is_mix_copy(stmt):
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    func = stmt.value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "copy"
        and "mixes[i, :]" in ast.unparse(stmt)
    )


def ast_proof() -> None:
    mhc_src = open(MHC_PATH).read()
    model_src = open(MODEL_PATH).read()
    mhc_tree = ast.parse(mhc_src)
    ast.parse(model_src)

    off_args, off_body = _prim_with_body(mhc_tree, "hc_split_sinkhorn_kernel")
    on_args, on_body = _prim_with_body(mhc_tree, "hc_split_sinkhorn_fused_kernel")

    # 1) the fused prim_func adds exactly one arg, `rsqrt`, right after `mixes`,
    #    declared as a 1-D fp32 tensor ([n], so rsqrt[i] is a scalar).
    assert off_args == ["mixes", "hc_scale", "hc_base", "pre", "post", "comb"], off_args
    assert on_args == [
        "mixes", "rsqrt", "hc_scale", "hc_base", "pre", "post", "comb"
    ], on_args
    fused_inner = next(
        n
        for n in next(
            o for o in mhc_tree.body
            if isinstance(o, ast.FunctionDef)
            and o.name == "hc_split_sinkhorn_fused_kernel"
        ).body
        if isinstance(n, ast.FunctionDef)
    )
    rsqrt_ann = next(
        a.annotation for a in fused_inner.args.args if a.arg == "rsqrt"
    )
    assert "T.Tensor[(n,), FP32]" in ast.unparse(rsqrt_ann), ast.unparse(rsqrt_ann)

    # 2) prefix (allocs incl. `mixes_shared = T.alloc_shared(mix_hc, FP32)` and
    #    `T.copy(mixes[i, :], mixes_shared)`) is byte-identical.
    k_off = [i for i, s in enumerate(off_body) if _is_mix_copy(s)]
    k_on = [i for i, s in enumerate(on_body) if _is_mix_copy(s)]
    assert len(k_off) == len(k_on) == 1, (k_off, k_on)
    k = k_off[0]
    assert _unparse(off_body[: k + 1]) == _unparse(on_body[: k + 1]), "prefix differs"
    assert "T.alloc_shared(mix_hc, FP32)" in _unparse(off_body[: k + 1]), (
        "fp32 shared landing alloc missing"
    )

    # 3) the fusion block is exactly two statements: `rs = rsqrt[i]` and a
    #    `for j in T.Parallel(mix_hc): mixes_shared[j] = mixes_shared[j] * rs`.
    #    It contains NO sigmoid/exp/reduce/scale/base -- the mul is isolated
    #    from the downstream expression, so it cannot become an FMA/reorder.
    block = on_body[k + 1 : k + 3]
    assert len(block) == 2
    assert ast.unparse(block[0]) == "rs = rsqrt[i]", ast.unparse(block[0])
    assert isinstance(block[1], ast.For) and "T.Parallel(mix_hc)" in ast.unparse(
        block[1].iter
    ), ast.unparse(block[1])
    assert len(block[1].body) == 1
    loop_stmt = ast.unparse(block[1].body[0])
    assert loop_stmt == "mixes_shared[j] = mixes_shared[j] * rs", loop_stmt
    for forbidden in ("sigmoid", "exp", "reduce", "hc_scale", "hc_base"):
        assert forbidden not in _unparse(block), f"fusion block leaked {forbidden}"

    # 4) the ENTIRE sigmoid/comb/reduce/sinkhorn suffix is byte-identical.
    assert _unparse(off_body[k + 1 :]) == _unparse(on_body[k + 3 :]), (
        "sinkhorn math suffix differs"
    )

    # 5) OFF arm in the call site is the untouched current chain.
    assert OFF_LINE in model_src, "OFF arm line changed"
    # gate: default-off per-call env read, HIP/non-XPU scoped.
    for frag in (
        "def _mhc_fused_sinkhorn_on_hip",
        f'os.environ.get("{ENV}", "0") != "0"',
        "_is_hip",
        "not _is_xpu",
        "hc_split_sinkhorn_fused(",
    ):
        assert frag in model_src, frag
    # module: default-off fuse_enabled + fallback guards.
    for frag in (
        f'ENV_MHC_FUSED_SINKHORN = "{ENV}"',
        "def fuse_enabled(",
        'os.environ.get(name, "0") != "0"',
        "def hc_split_sinkhorn_fused(",
        "is_gfx1250_supported()",
        "return hc_split_sinkhorn(",
    ):
        assert frag in mhc_src, frag

    print("AST proof: fused kernel == shipped kernel + 2-stmt fp32-landing block")
    print("           (rs = rsqrt[i]; mixes_shared[j] = mixes_shared[j] * rs)")
    print("AST proof: sinkhorn sigmoid/comb/reduce suffix byte-identical (AST)")
    print("AST proof: OFF arm byte-identical; env default-off per-call; fallback present")


# ---------------------------------------------------------------------------
# fp32 scalar element-by-element simulation (GPU-free)
# ---------------------------------------------------------------------------


def scalar_sim() -> int:
    """Emulate the one arithmetic that matters: the landed ``raw*rsqrt`` mul.

    OFF (torch): ``mixes = GEMM*rsqrt`` is one fp32 mul rounded to fp32, then
    fed to the shipped sinkhorn.  ON (fused): ``mixes_shared[j] =
    mixes_shared[j]*rs`` is the SAME standalone fp32 mul, landed in fp32 shared
    memory, then fed to the SAME sinkhorn.  Both therefore start the downstream
    math from bit-identical ``m``; a representative downstream ``sigmoid`` on
    that ``m`` is bit-identical too.

    A contrafactual "no-landing" arm (``raw*rs`` kept at f64 precision, as if
    the compiler folded the mul into ``m*scale+base`` / reassociated it) is
    computed to show the fp32 landing is LOAD-BEARING -- that is exactly the
    residual FMA/reorder risk the GPU ``torch.equal`` must rule out.

    The FULL sinkhorn (exp / reduce / sinkhorn_iters) is NOT simulated bit-for-
    bit here (numpy/CPU reductions != TileLang reductions); its equality is
    covered by the AST "suffix identical" exhibit + the GPU torch.equal gate.
    """
    rng = random.Random(20261108)
    scale0, base0 = 0.73, -0.41  # representative fp32 hc_scale[0], hc_base[0]

    raw_vals = [0.0, 1.0, -1.0, 1e-8, 1e-4, 8.0, -8.0, 0.5, -0.5]
    rs_vals = [1e-6, 1.0, 0.5, 2.0, 1e-3, 3.0]
    cases = [(f32(r), f32(s)) for r in raw_vals for s in rs_vals]
    cases += [
        (f32(rng.uniform(-8.0, 8.0)), f32(rng.uniform(1e-4, 3.0)))
        for _ in range(50000)
    ]

    landed_ok = 0
    no_landing_flips = 0
    for raw, rs in cases:
        # OFF and ON: one standalone fp32 multiply, rounded exactly once.
        m_off = f32(raw * rs)
        m_on = f32(raw * rs)
        assert m_off == m_on, (raw, rs, m_off, m_on)
        # Representative downstream (identical code in both arms -> same m).
        arg_off = f32(f32(m_off * scale0) + base0)
        arg_on = f32(f32(m_on * scale0) + base0)
        pre_off = sigmoid(arg_off)
        pre_on = sigmoid(arg_on)
        assert pre_off == pre_on, (raw, rs, pre_off, pre_on)
        landed_ok += 1

        # Contrafactual: `raw*rs` NOT rounded to fp32 before the downstream
        # mul-add (as if the compiler folded / reassociated it into an FMA,
        # keeping the product at higher precision).  If this matched, the
        # landing would be a no-op; it does not, so GPU confirmation that
        # TileLang honored the landing is mandatory.
        arg_bad = f32(raw * rs * scale0 + base0)  # no intermediate fp32 mul
        pre_bad = sigmoid(arg_bad)
        if pre_bad != pre_off:
            no_landing_flips += 1

    print(
        f"scalar sim: {landed_ok} (raw, rs) pairs -> OFF/ON landed mul + "
        "downstream pre bit-identical (2 arms)"
    )
    print(
        f"scalar sim: contrafactual (no fp32 landing) flips pre on "
        f"{no_landing_flips}/{landed_ok} cases -> landing is load-bearing; "
        "TileLang FMA/reorder MUST be ruled out by GPU torch.equal"
    )
    return landed_ok


# ---------------------------------------------------------------------------
# GPU branch: real kernel A/B + torch.equal + launch count + timing
# ---------------------------------------------------------------------------


def gpu_main() -> None:
    import torch
    import triton  # noqa: F401  (import guard)
    import tilelang  # noqa: F401  (TileLang backend for both kernels)

    sys.path.insert(0, SRC)
    from sglang.kernels.ops.layernorm.mhc import (
        hc_split_sinkhorn,
        hc_split_sinkhorn_fused,
    )

    hc_mult = 4
    mix_hc = (2 + hc_mult) * hc_mult
    sinkhorn_iters = 20
    eps = 1e-6
    # c=1 with DSPARK gamma=3 -> M = 4*bs speculative-verify rows; 4 is that
    # threshold.  Also cover chunk-boundary / small / large decode+prefill M.
    shapes = [1, 4, 8, 16, 18, 32, 64, 128, 256, 512, 2048, 4096]

    hc_scale = torch.randn((3,), dtype=torch.float32, device="cuda")
    hc_base = torch.randn((mix_hc,), dtype=torch.float32, device="cuda")

    # VERIFY_SEED lets the same GPU branch be re-run with independent data to
    # stress the residual TileLang FMA/reorder doubt across many random draws.
    seed = int(os.environ.get("VERIFY_SEED", "20261108"))
    torch.manual_seed(seed)
    print(f"VERIFY_SEED={seed}")

    def make(m):
        mixes_raw = torch.randn((m, mix_hc), dtype=torch.float32, device="cuda")
        rsqrt = torch.rand((m, 1), dtype=torch.float32, device="cuda") + 0.1
        return mixes_raw, rsqrt

    def arm_off(mixes_raw, rsqrt):
        mixes = (mixes_raw * rsqrt).unsqueeze(1)
        return hc_split_sinkhorn(
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
        )

    def arm_on(mixes_raw, rsqrt):
        return hc_split_sinkhorn_fused(
            mixes_raw, rsqrt, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
        )

    # Prove the ON arm actually dispatches the fused TileLang kernel and does
    # NOT silently take the wrapper's OFF fallback.  A fallback would replay the
    # identical OFF chain and make torch.equal a FALSE PASS -- exactly the kind
    # of "hit the wrong point" trap this project has been burned by before.
    import sglang.kernels.ops.layernorm.mhc as _mhc_mod

    _orig_off = _mhc_mod.hc_split_sinkhorn
    _orig_kern = _mhc_mod.hc_split_sinkhorn_fused_kernel
    _fallback_hits = [0]
    _fused_hits = [0]

    def _off_spy(*a, **k):
        _fallback_hits[0] += 1
        return _orig_off(*a, **k)

    def _kern_spy(*a, **k):
        _fused_hits[0] += 1
        return _orig_kern(*a, **k)

    _probe_raw, _probe_rs = make(8)
    _mhc_mod.hc_split_sinkhorn = _off_spy
    _mhc_mod.hc_split_sinkhorn_fused_kernel = _kern_spy
    try:
        arm_on(_probe_raw, _probe_rs)
        torch.cuda.synchronize()
    finally:
        _mhc_mod.hc_split_sinkhorn = _orig_off
        _mhc_mod.hc_split_sinkhorn_fused_kernel = _orig_kern
    assert _fused_hits[0] == 1 and _fallback_hits[0] == 0, (
        f"ON arm did NOT dispatch the fused kernel (fused={_fused_hits[0]}, "
        f"fallback={_fallback_hits[0]}) -- torch.equal would be a FALSE PASS"
    )
    print(
        f"ON-arm dispatch proven: fused kernel hits={_fused_hits[0]}, "
        f"fallback hits={_fallback_hits[0]}, "
        f"is_gfx1250_supported={_mhc_mod.is_gfx1250_supported()}"
    )

    cases = 0
    for m in shapes:
        for _ in range(3):
            mixes_raw, rsqrt = make(m)
            torch.cuda.synchronize()
            pre_off, post_off, comb_off = arm_off(mixes_raw, rsqrt)
            pre_on, post_on, comb_on = arm_on(mixes_raw, rsqrt)
            torch.cuda.synchronize()
            for name, a, b in (
                ("pre", pre_off, pre_on),
                ("post", post_off, post_on),
                ("comb", comb_off, comb_on),
            ):
                assert a.shape == b.shape, (m, name, a.shape, b.shape)
                assert torch.equal(a, b), (
                    f"BIT MISMATCH M={m} tensor={name} "
                    f"neq={int((a != b).sum())}/{a.numel()} "
                    f"nan_off={int(torch.isnan(a).sum())} nan_on={int(torch.isnan(b).sum())} "
                    f"inf_off={int(torch.isinf(a).sum())} inf_on={int(torch.isinf(b).sum())}"
                )
            cases += 1
    print(
        f"GPU torch.equal: PASS ({cases} cases, M in {shapes}, "
        "pre/post/comb whole-tensor bit compare)"
    )
    print("  M=4 is the c=1 M=4*bs speculative-verify threshold")

    # Launch count: OFF owns the extra `* rsqrt` mul launch; ON folds it in.
    try:
        from torch.profiler import ProfilerActivity, profile

        def cuda_launches(fn):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                fn()
            n = 0
            for e in prof.events():
                dt = getattr(e, "device_type", None)
                if dt is not None and "CUDA" in str(dt):
                    n += 1
            return n

        mixes_raw, rsqrt = make(8)
        arm_off(mixes_raw, rsqrt)  # warmup / JIT compile both kernels
        arm_on(mixes_raw, rsqrt)
        torch.cuda.synchronize()
        n_off = cuda_launches(lambda: arm_off(mixes_raw, rsqrt))
        n_on = cuda_launches(lambda: arm_on(mixes_raw, rsqrt))
        print(
            f"CUDA launches per call: OFF={n_off} ON={n_on} "
            f"saved={n_off - n_on} (expect >=1)"
        )
        if n_off - n_on < 1:
            print("WARNING: launch did not drop -- fusion target may not be hit")
    except Exception as exc:  # profiler backends vary across ROCm stacks
        print(f"CUDA launch count unavailable on this stack: {exc!r}")

    # Timing interface (CUDA events, eager so the removed launch is visible).
    iters = 200
    mixes_raw, rsqrt = make(8)
    arm_off(mixes_raw, rsqrt)
    arm_on(mixes_raw, rsqrt)
    torch.cuda.synchronize()

    def timed(fn):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters * 1000.0  # us / call

    off_us = [timed(lambda: arm_off(mixes_raw, rsqrt)) for _ in range(5)]
    on_us = [timed(lambda: arm_on(mixes_raw, rsqrt)) for _ in range(5)]
    print(
        f"time/call (median of 5 x {iters}): "
        f"OFF={statistics.median(off_us):.1f}us ON={statistics.median(on_us):.1f}us"
    )
    print("GPU bit-exactness MEASURED (torch.equal). No claim beyond the shapes above.")


# ---------------------------------------------------------------------------
# mode dispatch
# ---------------------------------------------------------------------------


def local_main() -> int:
    print("=" * 74)
    print("本机无 GPU / 无 triton / hip=None：仅 AST + fp32 标量逐元素仿真")
    print("未在 GPU 实测位精确 —— 上线前必须在集群跑本脚本 GPU 分支")
    print("=" * 74)
    ast_proof()
    n = scalar_sim()
    print(f"local: {n} scalar pairs simulated; landed mul bit-identical")
    print("=" * 74)
    print("未在 GPU 实测位精确（TileLang 编译器行为需 GPU torch.equal 坐实）")
    print("=" * 74)
    return 0


def main() -> int:
    have_gpu = have_triton = have_tilelang = False
    try:
        import torch

        have_gpu = torch.cuda.is_available()
    except Exception:
        pass
    try:
        import triton  # noqa: F401

        have_triton = True
    except Exception:
        pass
    try:
        import tilelang  # noqa: F401

        have_tilelang = True
    except Exception:
        pass
    if have_gpu and have_triton and have_tilelang:
        gpu_main()
        return 0
    print(
        f"[mode] local (cuda_available={have_gpu}, triton={have_triton}, "
        f"tilelang={have_tilelang}, hip=None on this host)"
    )
    return local_main()


if __name__ == "__main__":
    sys.exit(main())
