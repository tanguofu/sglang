#!/usr/bin/env python3
"""Fix: fused_store_index_cache.cuh pack_fp8 function uses CUDA-specific
fp8x2_e4m3_t{fp32x2_t{...}} constructor which doesn't exist on HIP.

On HIP, fp8x2_e4m3_t is unsigned short, and there's no implicit conversion
from fp32x2_t (HIP_vector_type<float, 2>).

Fix: Use HIP's __hip_cvt_float2_to_fp8x2 conversion function on ROCm.
"""
import os

FILE = "/sgl-workspace/sglang/python/sglang/jit_kernel/csrc/dsa/fused_store_index_cache.cuh"

with open(FILE, "r") as f:
    content = f.read()

old = """[[maybe_unused]]
SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {
  return fp8x2_e4m3_t{fp32x2_t{fp8_e4m3_clip(x), fp8_e4m3_clip(y)}};
}"""

new = """[[maybe_unused]]
SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {
#ifdef USE_ROCM
  return __hip_cvt_float2_to_fp8x2(
      make_float2(fp8_e4m3_clip(x), fp8_e4m3_clip(y)),
      __HIP_SATFINITE, __HIP_E4M3);
#else
  return fp8x2_e4m3_t{fp32x2_t{fp8_e4m3_clip(x), fp8_e4m3_clip(y)}};
#endif
}"""

if old in content:
    content = content.replace(old, new, 1)
    with open(FILE, "w") as f:
        f.write(content)
    print("[OK] Patched pack_fp8 with HIP conversion function")
elif "__hip_cvt_float2_to_fp8x2" in content:
    print("[SKIP] pack_fp8 already patched with HIP conversion")
else:
    print("[ERROR] pack_fp8 pattern not found")
    import sys; sys.exit(1)
