#!/usr/bin/env python3
"""Port patch 3.1 (deepseek_nextn alt_stream on HIP) to post1 deepseek_nextn.py.

Semantic change: the alt_stream (multi-stream) is created on CUDA or NPU-multi-stream.
On HIP (ROCm) we also need it, so widen the gate to include is_hip().

Base line:
    if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
Branch line:
    if _is_cuda or is_hip() or envs.SGLANG_NPU_USE_MULTI_STREAM.get()

Also add `is_hip` to the platform import (from sglang.srt.utils).

Idempotent.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/port-post1/base/srt_models_deepseek_nextn.py")
src = path.read_text()
changed = []

# 1) import is_hip
imp_old = "from sglang.srt.utils import BumpAllocator, add_prefix, is_cuda, is_npu"
imp_new = "from sglang.srt.utils import BumpAllocator, add_prefix, is_cuda, is_hip, is_npu"
if "is_hip" in src.split("\n")[[l for l, line in enumerate(src.splitlines()) if line.startswith("from sglang.srt.utils import")][0]] if False else "":
    pass
if ", is_hip," in src:
    changed.append("import is_hip: skipped")
else:
    assert imp_old in src, "3.1: import anchor not found"
    src = src.replace(imp_old, imp_new, 1)
    changed.append("import is_hip: applied")

# 2) widen alt_stream gate
gate_old = "if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get()"
gate_new = "if _is_cuda or is_hip() or envs.SGLANG_NPU_USE_MULTI_STREAM.get()"
if "is_hip() or envs.SGLANG_NPU_USE_MULTI_STREAM" in src:
    changed.append("alt_stream gate: skipped")
else:
    assert gate_old in src, "3.1: alt_stream gate not found"
    src = src.replace(gate_old, gate_new, 1)
    changed.append("alt_stream gate: applied")

path.write_text(src)
print(f"[ok] {path}: " + ", ".join(changed))
