#!/usr/bin/env python3
"""Port patch for forward_mha.py: guard the aiter+HIP branch against non-power-of-2 dims.

Problem: GLM-5.2 has qk_nope_head_dim=192 (not a power of 2). The _is_hip+aiter
branch in _concat_and_cast_mha_k unconditionally calls concat_and_cast_mha_k_triton,
which uses tl.arange(0, nope_dim). Triton's arange requires a power-of-2 range,
so nope_dim=192 raises "arange's range must be a power of 2" at kernel compile time.

Fix: add a power-of-2 guard before calling the triton kernel. When nope_dim or
rope_dim is not a power of 2, fall back to plain slice assignment (the same code
path used by the else branch).

Idempotent: detects the guard marker and skips if already applied.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/port-post1/forward_mha.py")
src = path.read_text()

GUARD_MARKER = "_np2_aiter_guard"

old = (
    '        elif _is_hip and self.current_attention_backend == "aiter":\n'
    "            k = k_nope.new_empty(*k_shape)\n"
    "            concat_and_cast_mha_k_triton(k, k_nope, k_pe)"
)

new = (
    '        elif _is_hip and self.current_attention_backend == "aiter":  # '
    + GUARD_MARKER
    + "\n"
    "            k = k_nope.new_empty(*k_shape)\n"
    "            _np2 = (self.qk_nope_head_dim & (self.qk_nope_head_dim - 1)) == 0\n"
    "            _rp2 = (self.qk_rope_head_dim & (self.qk_rope_head_dim - 1)) == 0\n"
    "            if _np2 and _rp2:\n"
    "                concat_and_cast_mha_k_triton(k, k_nope, k_pe)\n"
    "            else:\n"
    "                k[..., : self.qk_nope_head_dim] = k_nope\n"
    "                k[..., self.qk_nope_head_dim :] = k_pe"
)

if GUARD_MARKER in src:
    print(f"[forward_mha] already patched ({GUARD_MARKER} present), skipped")
    sys.exit(0)

if old not in src:
    print("[forward_mha] WARN: patch target not found (already patched or source changed)")
    # Exit 0 so the Dockerfile RUN does not fail on idempotent re-application
    sys.exit(0)

n = src.count(old)
src = src.replace(old, new, 1)
path.write_text(src)
print(f"[forward_mha] patched: aiter+HIP branch now guards non-power-of-2 dims ({n} site)")
