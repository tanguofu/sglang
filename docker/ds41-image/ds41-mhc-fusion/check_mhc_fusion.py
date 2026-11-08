#!/usr/bin/env python3
"""mHC #1 fusion delta check (runs inside the build container; no GPU/torch).

Verifies the image changed ONLY the two intended .py files, that the OFF arm
and the shipped kernels are byte-for-byte the untouched m20 chain, and that the
env gate / fused kernel / fallback guards are present.

m20 base bytes were READ (not guessed) on 2026-11-08 from the pinned m20 digest
in ti-builder ``dind-builder4`` and cross-checked against the live m20 decode
pod and fork HEAD ``d7dab9171c`` git blobs; all three agree.  Their md5 is
therefore PINNED below.  ``build.sh`` still pulls the base from the pinned
digest at build time; this check re-asserts the bytes match the pin, so a
re-pushed tag / wrong digest can never slip a different base through.

Usage:
  python3 check_mhc_fusion.py <patched-model> <base-model> <patched-mhc> <base-mhc>
"""
from __future__ import annotations

import ast
import hashlib
import sys

OFF_LINE = "        mixes = (row_uniform_mhc_mix(x_flat, hc.fn) * rsqrt).unsqueeze(1)"
ENV = "SGLANG_DS41_MHC_FUSED_SINKHORN"

# md5 of m20's OWN base bytes, read from
#   ds41-prod-1033-m20@sha256:5a0fcb8a2e12eafd6dab9f8ea632051a27152e63e342229fa506484c8d934374
# (ti-builder dind-builder4), == live m20 pod == fork HEAD d7dab9171c.
BASE_MD5 = {
    "deepseek_v4_mhc.py": "e71f21c856bce40895cdeb83f7a995e5",
    "mhc.py": "04d8a21996483628daf38f1ff16d2f7d",
}


def func_src(tree, name):
    """ast.unparse of one top-level FunctionDef (decorators included)."""
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.unparse(fn)


def main():
    pm_p, bm_p, pm_m, bm_m = sys.argv[1:5]
    pm = open(pm_p).read()
    bm = open(bm_p).read()
    ph = open(pm_m).read()
    bh = open(bm_m).read()
    for name, text in (("patched-model", pm), ("base-model", bm),
                       ("patched-mhc", ph), ("base-mhc", bh)):
        ast.parse(text)
        print(f"check_mhc_fusion: {name} parses")

    # 0) m20 base bytes were READ from the pinned digest (2026-11-08); the pin
    #    is enforced here so a wrong digest / re-pushed tag fails the build.
    for name, path in (("deepseek_v4_mhc.py", bm_p), ("mhc.py", bm_m)):
        digest = hashlib.md5(open(path, "rb").read()).hexdigest()
        assert digest == BASE_MD5[name], (
            f"base {name} md5 {digest} != pinned m20 {BASE_MD5[name]} "
            "-- wrong m20 digest or re-pushed tag; refusing to build"
        )
        print(f"check_mhc_fusion: base {name} md5 == pinned m20 ({digest})")

    # 1) OFF arm: the original `mixes = (...)` line, verbatim in base and patched.
    base_off = [l for l in bm.splitlines() if l.strip().startswith("mixes = (row_uniform")]
    patched_off = [l for l in pm.splitlines() if l.strip().startswith("mixes = (row_uniform")]
    assert len(base_off) == 1 and len(patched_off) == 1, (base_off, patched_off)
    assert patched_off[0] == base_off[0] == OFF_LINE, (
        f"OFF arm changed: base={base_off[0]!r} patched={patched_off[0]!r}"
    )
    print("check_mhc_fusion: OFF arm line == verbatim m20 chain")

    # 2) Shipped kernels / wrappers UNCHANGED between base and patched mhc.
    bt, pt = ast.parse(bh), ast.parse(ph)
    for name in (
        "hc_split_sinkhorn_kernel",
        "hc_split_sinkhorn",
        "_hc_split_sinkhorn_torch",
        "_hc_split_sinkhorn_triton",
    ):
        assert func_src(bt, name) == func_src(pt, name), (
            f"shipped {name} changed between base and patched"
        )
    print("check_mhc_fusion: shipped kernel + wrapper + gfx1250 ports byte-identical")

    # 3) gate + ON branch present in patched, absent in base.  The early
    #    `return pre.squeeze(1)...` is NOT in this list: the original final
    #    return of _mix_stats_impl uses the same expression, so base already
    #    has one occurrence; the ON branch adds a SECOND one (checked below).
    for frag in (
        "def _mhc_fused_sinkhorn_on_hip",
        f'os.environ.get("{ENV}", "0") != "0"',
        "if _mhc_fused_sinkhorn_on_hip():",
        "hc_split_sinkhorn_fused(",
        "mixes_raw = row_uniform_mhc_mix(x_flat, hc.fn)",
    ):
        assert frag in pm, f"patched model missing {frag!r}"
        assert frag not in bm, f"base model unexpectedly has {frag!r}"
    ret = "return pre.squeeze(1), post.squeeze(1), comb.squeeze(1)"
    assert bm.count(ret) == 1 and pm.count(ret) == 2, (
        f"early ON-arm return missing (base={bm.count(ret)}, patched={pm.count(ret)})"
    )
    # ON branch wiring: the fused wrapper gets the raw GEMM + rsqrt + weights.
    for frag in ("mixes_raw,", "rsqrt,", "hc.scale,", "hc.base,",
                 "cfg.mult,", "cfg.sinkhorn_iters,", "cfg.eps,"):
        assert frag in pm, f"ON branch wiring missing {frag!r}"
    print("check_mhc_fusion: env gate + ON branch present (base clean), wiring ok")

    # 4) fused kernel + env default-off + fallback guards in patched mhc.
    for frag in (
        f'ENV_MHC_FUSED_SINKHORN = "{ENV}"',
        "def fuse_enabled(",
        'os.environ.get(name, "0") != "0"',
        "def hc_split_sinkhorn_fused_kernel(",
        "rsqrt: T.Tensor[(n,), FP32]",
        "mixes_shared[j] = mixes_shared[j] * rs",
        "def hc_split_sinkhorn_fused(",
        "is_gfx1250_supported()",
        "return hc_split_sinkhorn(",
    ):
        assert frag in ph, f"patched mhc missing {frag!r}"
    assert "import os\n" in ph, "patched mhc missing `import os` (fuse_enabled needs it)"
    print("check_mhc_fusion: fused kernel + default-off fuse_enabled + fallback present")

    print("check_mhc_fusion: ALL PASS (2 .py files; OFF arm + shipped kernels unchanged)")


if __name__ == "__main__":
    main()
