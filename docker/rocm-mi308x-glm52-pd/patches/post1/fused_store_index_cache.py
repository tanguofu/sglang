#!/usr/bin/env python3
"""Port patch 05/6.1 (fused_store_index_cache.cuh pack_fp8 on ROCm) to post1.

Problem: the local pack_fp8 uses `fp8x2_e4m3_t{fp32x2_t{...}}`, which does not compile
on ROCm/gfx942 (fp8x2_e4m3_t is unsigned short, no fp32x2_t constructor).

Fix (aligned with how post1's own deepseek_v4 kernels do it): include
sgl_kernel/deepseek_v4/fp8_utils.cuh and `using deepseek_v4::fp8::pack_fp8`.
That header has a correct software pack_fp8 for ROCm (and the CUDA ctor path
otherwise), so we drop the local pack_fp8 definition and the cuda_fp8.h include
(fp8_utils.cuh manages its own fp8 include).

Idempotent.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/port-post1/base/jit_kernel_csrc_dsa_fused_store_index_cache.cuh")
src = path.read_text()
changed = []

# 1) add include for fp8_utils.cuh after the sgl_kernel includes block
inc_anchor = "#include <sgl_kernel/warp.cuh>\n"
inc_new = inc_anchor + "#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>\n"
if "deepseek_v4/fp8_utils.cuh" in src:
    changed.append("include: skipped")
else:
    assert inc_anchor in src, "05: include anchor not found"
    src = src.replace(inc_anchor, inc_new, 1)
    changed.append("include: applied")

# 2) drop the local cuda_fp8.h include (fp8_utils.cuh manages fp8 include via
#    its own #ifndef USE_ROCM guard). Keep the <cstdint>/<bit> std includes.
if "#include <cuda_fp8.h>\n" in src:
    src = src.replace("#include <cuda_fp8.h>\n", "", 1)
    changed.append("drop cuda_fp8.h: applied")
else:
    changed.append("drop cuda_fp8.h: skipped")

# 3) replace the local pack_fp8 definition with a using-declaration.
local_def = (
    "[[maybe_unused]]\n"
    "SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {\n"
    "  return fp8x2_e4m3_t{fp32x2_t{fp8_e4m3_clip(x), fp8_e4m3_clip(y)}};\n"
    "}\n"
)
using_decl = "using deepseek_v4::fp8::pack_fp8;\n"
if using_decl in src and local_def not in src:
    changed.append("pack_fp8: skipped")
else:
    assert local_def in src, "05: local pack_fp8 def not found"
    src = src.replace(local_def, using_decl, 1)
    changed.append("pack_fp8: applied")

path.write_text(src)
print(f"[ok] {path}: " + ", ".join(changed))
