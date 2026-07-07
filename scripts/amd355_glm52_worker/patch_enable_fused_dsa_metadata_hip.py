"""
Patch: Enable fused DSA metadata generation on HIP (ROCm)

The fused_dsa_decode_metadata, fused_dsa_target_verify_metadata, and
fused_dsa_draft_extend_metadata Triton kernels are pure Triton with no
CUDA-specific primitives, but are gated on `is_cuda() and not _is_hip`.

On HIP, the fallback path does individual copy_() operations for:
  - cache_seqlens
  - cu_seqlens_k (via torch.cumsum + copy_)
  - page_table_1 (via indexing + copy_)
  - dsa_cache_seqlens (via compute_dsa_seqlens + copy_)

These run per-layer (78 layers), adding overhead.
The fused Triton kernel does all of this in a single kernel launch.

This patch changes `is_cuda() and not _is_hip` to `is_cuda()` for the
three fused metadata generation paths. On ROCm, is_cuda() returns True
(PyTorch ROCm uses CUDA backend), so the fused path will be enabled.

Note: The fused_metadata_copy_cuda path (line ~1554) uses a CUDA-specific
JIT kernel and is NOT touched by this patch.
"""

import sys
from pathlib import Path


def patch_file(filepath: str) -> bool:
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] File not found: {filepath}")
        return False

    content = path.read_text()
    original = content
    count = 0

    # The three fused metadata generation guards use: `if is_cuda() and not _is_hip:`
    # We change them to: `if is_cuda():`
    # This enables the Triton-based fused metadata kernels on HIP.
    old = "if is_cuda() and not _is_hip:"
    new = "if is_cuda():"

    while old in content:
        content = content.replace(old, new, 1)
        count += 1

    if count > 0:
        print(f"[OK] Replaced {count} occurrences of `is_cuda() and not _is_hip:` -> `is_cuda():`")
    else:
        print("[WARN] No occurrences found - may already be patched")

    if content != original:
        path.write_text(content)
        print(f"[DONE] Patched {filepath}")
        return True
    else:
        print(f"[SKIP] No changes needed for {filepath}")
        return False


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else \
        "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"
    patch_file(target)
