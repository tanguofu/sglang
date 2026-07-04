#!/usr/bin/env python3
"""Fix: Replace #include <cuda_fp8.h> with HIP-compatible include in JIT kernel sources.

The DSA indexer fusion JIT kernel (fused_store_index_cache.cuh) includes
<cuda_fp8.h> which doesn't exist on ROCm. This causes JIT compilation to
fail, falling back to a slower non-fused path.

Fix: Use platform-conditional include. -DUSE_ROCM is already passed by the
build system, so we can check for it.
"""
import os, glob

# Find all .cuh and .cu files that include cuda_fp8.h
SEARCH_DIRS = [
    "/sgl-workspace/sglang/python/sglang/jit_kernel/csrc",
    "/sgl-workspace/sglang/python/sglang/jit_kernel/include",
]

old_include = "#include <cuda_fp8.h>"
new_include = """#ifdef USE_ROCM
#include <hip/hip_fp8.h>
#else
#include <cuda_fp8.h>
#endif"""

patched_count = 0
for search_dir in SEARCH_DIRS:
    if not os.path.isdir(search_dir):
        continue
    for root, dirs, files in os.walk(search_dir):
        for fname in files:
            if not (fname.endswith(".cuh") or fname.endswith(".cu") or fname.endswith(".cuh.in")):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r") as f:
                    content = f.read()
            except Exception:
                continue
            if old_include in content:
                content = content.replace(old_include, new_include, 1)
                with open(fpath, "w") as f:
                    f.write(content)
                print(f"[OK] Patched: {fpath}")
                patched_count += 1
            elif "hip/hip_fp8.h" in content:
                print(f"[SKIP] Already patched: {fpath}")

if patched_count == 0:
    print("[INFO] No files needed patching (may already be patched)")
else:
    print(f"\n[DONE] Patched {patched_count} file(s)")
