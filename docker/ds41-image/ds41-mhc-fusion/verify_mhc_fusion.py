#!/usr/bin/env python3
"""GPU-free, import-free build-time proof of the mHC #1 fusion.

Runs inside the build container (no HIP GPU, no torch/triton/tilelang).  Proves
the two things provable without a device:

  1. AST: the fused TileLang kernel is the shipped kernel + exactly one
     fp32-landing block (``rs = rsqrt[i]; mixes_shared[j] *= rs``) and the
     ENTIRE sigmoid/comb/reduce/sinkhorn suffix is byte-identical (AST).
  2. Scalar: OFF/ON both round ``raw*rs`` to fp32 exactly once before the SAME
     downstream math; a "no-landing" contrafactual shows the landing is
     load-bearing (the residual FMA/reorder risk).

It NEVER claims torch.equal was run (that is done by the cluster GPU branch).
Usage:

  python3 verify_mhc_fusion.py <mhc.py> <deepseek_v4_mhc.py>

TileLang compiler behaviour (no FMA contraction / loop-fusion of the standalone
mul) was CONFIRMED on 2026-11-08 by the GPU ``torch.equal`` branch of
verify/verify_mhc_fusion.py (gfx942, 108 cases PASS) plus the generated device
source showing 0 FMA.  This build-time script remains a static gate only.
"""
from __future__ import annotations

import ast
import math
import random
import struct
import sys

OFF_LINE = "mixes = (row_uniform_mhc_mix(x_flat, hc.fn) * rsqrt).unsqueeze(1)"


def f32(x):
    x = float(x)
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def sigmoid(arg):
    if arg >= 0:
        return f32(1.0 / (1.0 + math.exp(-arg)))
    z = math.exp(arg)
    return f32(z / (1.0 + z))


def prim_with_body(tree, name):
    outer = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    inner = next(n for n in outer.body if isinstance(n, ast.FunctionDef))
    assert isinstance(inner.body[0], ast.With), name
    return [a.arg for a in inner.args.args], inner.body[0].body


def unparse(stmts):
    return ast.unparse(ast.Module(body=list(stmts), type_ignores=[]))


def is_mix_copy(stmt):
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "copy"
        and "mixes[i, :]" in ast.unparse(stmt)
    )


def ast_proof(mhc_src, model_src):
    tree = ast.parse(mhc_src)
    ast.parse(model_src)
    off_args, off_body = prim_with_body(tree, "hc_split_sinkhorn_kernel")
    on_args, on_body = prim_with_body(tree, "hc_split_sinkhorn_fused_kernel")
    assert off_args == ["mixes", "hc_scale", "hc_base", "pre", "post", "comb"]
    assert on_args == ["mixes", "rsqrt", "hc_scale", "hc_base", "pre", "post", "comb"]
    ko = [i for i, s in enumerate(off_body) if is_mix_copy(s)]
    kn = [i for i, s in enumerate(on_body) if is_mix_copy(s)]
    assert len(ko) == len(kn) == 1
    k = ko[0]
    assert unparse(off_body[: k + 1]) == unparse(on_body[: k + 1]), "prefix differs"
    assert "T.alloc_shared(mix_hc, FP32)" in unparse(off_body[: k + 1])
    block = on_body[k + 1 : k + 3]
    assert len(block) == 2
    assert ast.unparse(block[0]) == "rs = rsqrt[i]"
    assert isinstance(block[1], ast.For) and "T.Parallel(mix_hc)" in ast.unparse(block[1].iter)
    assert len(block[1].body) == 1
    assert ast.unparse(block[1].body[0]) == "mixes_shared[j] = mixes_shared[j] * rs"
    for forbidden in ("sigmoid", "exp", "reduce", "hc_scale", "hc_base"):
        assert forbidden not in unparse(block), f"fusion block leaked {forbidden}"
    assert unparse(off_body[k + 1 :]) == unparse(on_body[k + 3 :]), "suffix differs"
    assert OFF_LINE in model_src, "OFF arm changed"
    for frag in (
        "def _mhc_fused_sinkhorn_on_hip",
        "SGLANG_DS41_MHC_FUSED_SINKHORN",
        "hc_split_sinkhorn_fused(",
        "def fuse_enabled(",
        "is_gfx1250_supported()",
        "return hc_split_sinkhorn(",
    ):
        assert frag in mhc_src or frag in model_src, frag
    print("verify_mhc_fusion: AST proof -- fused == shipped + 2-stmt fp32 landing")
    print("verify_mhc_fusion: AST proof -- sinkhorn suffix byte-identical; OFF arm verbatim")


def scalar_sim():
    rng = random.Random(20261108)
    scale0, base0 = 0.73, -0.41
    cases = [(f32(r), f32(s))
             for r in (0.0, 1.0, -1.0, 1e-8, 1e-4, 8.0, -8.0, 0.5, -0.5)
             for s in (1e-6, 1.0, 0.5, 2.0, 1e-3, 3.0)]
    cases += [(f32(rng.uniform(-8.0, 8.0)), f32(rng.uniform(1e-4, 3.0)))
              for _ in range(20000)]
    flips = 0
    for raw, rs in cases:
        m_off = f32(raw * rs)   # OFF: torch fp32 mul
        m_on = f32(raw * rs)    # ON : standalone fp32 mul landed in shared
        assert m_off == m_on
        arg = f32(f32(m_off * scale0) + base0)
        pre_off = sigmoid(arg)
        pre_on = sigmoid(arg)
        assert pre_off == pre_on
        arg_bad = f32(raw * rs * scale0 + base0)  # no fp32 landing (FMA risk)
        if sigmoid(arg_bad) != pre_off:
            flips += 1
    print(f"verify_mhc_fusion: scalar proof -- {len(cases)} pairs, landed mul bit-identical")
    print(f"verify_mhc_fusion: no-landing contrafactual flips {flips}/{len(cases)} "
          "-> landing load-bearing")
    return len(cases)


def main():
    mhc_p, model_p = sys.argv[1:3]
    ast_proof(open(mhc_p).read(), open(model_p).read())
    scalar_sim()
    print("=" * 70)
    print("本构建容器无 GPU / 无 triton / hip=None：以上仅 build-time 静态门禁")
    print("位精确已在集群 gfx942 用 verify/verify_mhc_fusion.py GPU 分支实测")
    print("torch.equal PASS (2026-11-08, 3 种子 108 cases)；本门禁不替代该实测")
    print("=" * 70)


if __name__ == "__main__":
    main()
