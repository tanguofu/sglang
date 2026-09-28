#!/usr/bin/env python3
"""Enable HiCache JIT kernel for GLM-5.2 MLA (element_size=576) on ROCm.

The JIT HiCache kernel (hicache.cuh) required element_size % 128 == 0, which
excluded GLM-5.2 MLA (kv_cache_dim=576, 576%128=64) and forced the slow AOT
fallback (8-byte loads). Relax the alignment to 64 bytes (576%64=0).

The 64-byte path uses PackageType<64/kNumThreads> = uint1 (4B) for
kNumThreads=16, vs uint2 (8B) for the 128-byte path — half the per-load
vector width, but still faster than the AOT fallback's scalar loop.

Idempotent: re-running on an already-patched file is a no-op.
"""
from __future__ import annotations

import sys
from pathlib import Path

# hicache.cuh: load_vec 128 -> 64
CUH_OLD = """  static_assert(kBytes % 128 == 0, "kBytes must be multiple of 128 bytes");
  static_assert(128 % kNumThreads == 0, "kNumThreads must divide 128 bytes");
  constexpr uint32_t kLoopCount = kBytes / 128;
  using Package = details::PackageType<128 / kNumThreads>;"""

CUH_NEW = """  // 64-byte granularity: 128 excluded GLM-5.2 MLA (576%128=64).
  static_assert(kBytes % 64 == 0, "kBytes must be multiple of 64 bytes");
  static_assert(64 % kNumThreads == 0, "kNumThreads must divide 64 bytes");
  constexpr uint32_t kLoopCount = kBytes / 64;
  using Package = details::PackageType<64 / kNumThreads>;"""

# hicache.py: can_use_hicache_jit_kernel 128 -> 64
PY_OLD = "    if element_size % 128 != 0:"
PY_NEW = "    if element_size % 64 != 0:"


def patch_file(path: Path, old: str, new: str, label: str) -> None:
    src = path.read_text()
    if new in src:
        print(f"[hicache-576] already patched: {label} {path}")
        return
    if old not in src:
        raise SystemExit(
            f"[hicache-576] could not find target block in {label} {path}; "
            "file may have been refactored — inspect manually"
        )
    path.write_text(src.replace(old, new, 1))
    print(f"[hicache-576] patched {label}: {path}")


def main() -> None:
    base = Path("/sgl-workspace/sglang/python/sglang")
    patch_file(
        base / "kernels/jit/csrc/kvcacheio/hicache.cuh", CUH_OLD, CUH_NEW, "hicache.cuh"
    )
    patch_file(
        base / "kernels/ops/kvcache/hicache.py", PY_OLD, PY_NEW, "hicache.py"
    )


if __name__ == "__main__":
    main()
