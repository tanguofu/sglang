#!/usr/bin/env python3
"""Stage the mHC #1 fusion image's two .py files from the REVIEWED WORKTREE.

m20's ``deepseek_v4_mhc.py`` / ``mhc.py`` bytes are pulled from the PINNED m20
DIGEST by ``build.sh``.  Those bytes were verified (2026-11-08) to be
byte-identical to the fork-HEAD base of this worktree from THREE independent
sources:

  * fork HEAD ``d7dab9171c`` git blobs,
  * the live m20 decode pod ``/sgl-workspace/sglang/python/sglang/...``,
  * ti-builder ``dind-builder4``, image
    ``ds41-prod-1033-m20@sha256:5a0fcb8a2e12eafd6dab9f8ea632051a27152e63e342229fa506484c8d934374``.

So each reviewed worktree file is exactly ``m20 base + insertion-only delta``.

The earlier AST-lifting patch re-derived that delta and silently dropped the
leading comment block and reordered ``hc_split_sinkhorn_fused``; the staged
bytes were therefore NOT byte-identical to what the GPU ``torch.equal`` was run
against.  This script instead ASSERTS the pulled base is a pure
line-subsequence of each reviewed file (difflib: only ``equal``/``insert``
opcodes, no ``delete``/``replace``) and then EMITS THE REVIEWED WORKTREE BYTES
verbatim.  The built image is thus byte-identical to the measured code.

Any divergence of the pulled m20 base (different digest, re-pushed tag) makes
the subsequence assertion FAIL LOUDLY; ``check_mhc_fusion.py`` additionally
pins the m20 base md5.  Nothing is silently emitted.

Usage: python3 patch_mhc_fusion.py <m20src-dir> <proto-root> <out-dir>
  <m20src-dir>  contains m20's OWN deepseek_v4_mhc.py and mhc.py
  <proto-root>  worktree root (contains python/sglang/... reviewed sources)
  <out-dir>     staged output (patched deepseek_v4_mhc.py + mhc.py)
"""

from __future__ import annotations

import ast
import difflib
import pathlib
import shutil
import sys

MODEL_REL = "python/sglang/srt/models/deepseek_v4_mhc.py"
MHC_REL = "python/sglang/kernels/ops/layernorm/mhc.py"


def assert_insertion_only(base_bytes: bytes, proto_bytes: bytes, label: str) -> None:
    """m20 base must be a pure insertion-subsequence of the reviewed file."""
    b = base_bytes.decode("utf-8").splitlines(keepends=True)
    p = proto_bytes.decode("utf-8").splitlines(keepends=True)
    opcodes = difflib.SequenceMatcher(None, b, p, autojunk=False).get_opcodes()
    bad = [op for op in opcodes if op[0] not in ("equal", "insert")]
    if bad:
        raise SystemExit(
            f"patch_mhc_fusion: {label}: pulled m20 base is NOT a pure "
            f"insertion-subsequence of the reviewed file (first non-insert "
            f"opcode {bad[0]}). m20 bytes drifted from the verified base -- "
            "re-review before building; refusing to emit a silently-wrong file."
        )
    reconstructed = "".join(
        chunk
        for tag, i1, i2, j1, j2 in opcodes
        for chunk in (b[i1:i2] if tag == "equal" else p[j1:j2])
    )
    assert reconstructed == proto_bytes.decode("utf-8"), (
        f"{label}: insertion-only reconstruction mismatch"
    )
    ast.parse(proto_bytes.decode("utf-8"))


def main() -> None:
    m20src, protoroot, out = map(pathlib.Path, sys.argv[1:4])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for name, rel in (("deepseek_v4_mhc.py", MODEL_REL), ("mhc.py", MHC_REL)):
        base_bytes = (m20src / name).read_bytes()
        proto_bytes = (protoroot / rel).read_bytes()
        assert_insertion_only(base_bytes, proto_bytes, name)
        # Emit the REVIEWED bytes verbatim (not a re-derived delta).  The GPU
        # torch.equal was measured against exactly these bytes.
        (out / name).write_bytes(proto_bytes)
        print(
            f"staged {name}: {len(proto_bytes.decode('utf-8').splitlines())} lines "
            "(byte-identical to reviewed worktree)"
        )
    print(
        "patch_mhc_fusion: m20 base is insertion-only ancestor; emitted reviewed "
        "worktree bytes verbatim"
    )


if __name__ == "__main__":
    main()
