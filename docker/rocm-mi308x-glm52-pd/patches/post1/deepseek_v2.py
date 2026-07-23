#!/usr/bin/env python3
"""Port patch 2.1 (PCG_DSV2_DUAL_STREAM HIP support) to post1 deepseek_v2.py.

Semantic change: widen the dual-stream enable gate from CUDA-only to CUDA+HIP:
    _is_cuda and envs.SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM.get()
  -> (_is_cuda or _is_hip) and envs.SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM.get()

Idempotent: skips if already patched.
"""
import sys
from pathlib import Path

OLD = "_is_cuda and envs.SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM.get()"
NEW = "(_is_cuda or _is_hip) and envs.SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM.get()"

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/port-post1/base/deepseek_v2.py")
src = path.read_text()

if NEW in src and OLD not in src:
    print(f"[skip] {path}: patch 2.1 already applied")
    sys.exit(0)

count = src.count(OLD)
if count == 0:
    print(f"[warn] {path}: target pattern not found (already patched or base changed?)", file=sys.stderr)
    sys.exit(0)

src = src.replace(OLD, NEW)
path.write_text(src)
print(f"[ok] {path}: patch 2.1 applied ({count} replacement)")
