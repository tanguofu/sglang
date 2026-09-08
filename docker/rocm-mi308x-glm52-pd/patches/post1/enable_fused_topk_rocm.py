#!/usr/bin/env python3
"""Enable fused DSA top-k on ROCm under PD disaggregation.

The `should_remap_pd_dsa_seed_to_local_slots` guard gates allocator-local fused
TopK on `is_cuda()` only, which excludes ROCm/HIP. The fused kernels
(`sgl_kernel.fast_topk_v2`, `fast_topk_transform_fused`) are available and
numerically correct on ROCm (verified: identical top-k set vs torch.topk), and
the ROCm MQA-logits path (`aiter_paged_mqa_logits`) already outputs float32, so
the dtype requirement of the fused kernel is satisfied.

Effect: without this fix, PD decode workers log
"Disabling fused DSA top-k for IndexShare under PD disaggregation" and fall back
to the unfused torch.topk path, which correlates with a lower MTP accept length
(~2.67 vs ~3.31 on non-PD builds).

Idempotent: re-running on an already-patched file is a no-op.
"""
from __future__ import annotations

import sys
from pathlib import Path

OLD = "        is_cuda()\n        and envs.SGLANG_DSA_FUSE_TOPK.get()"
NEW = "        (is_cuda() or is_hip())\n        and envs.SGLANG_DSA_FUSE_TOPK.get()"


def main(path: str) -> None:
    p = Path(path)
    src = p.read_text()
    if NEW in src:
        print(f"[fused-topk-roc] already patched: {p}")
        return
    if OLD not in src:
        raise SystemExit(
            f"[fused-topk-roc] could not find target block in {p}; "
            "file may have been refactored — inspect manually"
        )
    p.write_text(src.replace(OLD, NEW, 1))
    print(f"[fused-topk-roc] patched {p}: is_cuda() -> (is_cuda() or is_hip())")


if __name__ == "__main__":
    main(sys.argv[1])
