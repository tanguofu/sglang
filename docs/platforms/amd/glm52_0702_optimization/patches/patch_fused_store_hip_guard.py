#!/usr/bin/env python3
"""Fix: Prevent fused_store_index_cache JIT kernel from being used on HIP.

The fused store JIT kernel (fused_store_index_cache.cuh) uses per-warp scaling
(128 elements, one scale per warp), while AITER's indexer_k_quant_and_cache uses
a different scaling scheme (block_size + scale_fmt). The two schemes produce
different FP8 values, so using the fused store kernel on HIP changes K-cache
values, which changes attention scores, which degrades MTP accept rate
(81.6% → 71.79%).

Fix: Add _is_cuda check in _fused_k_prepare_and_store, so HIP always uses the
fallback path (fused_k_indexer_norm_rope + AITER indexer_k_quant_and_cache).
This preserves fusion benefits (fused Q prepare, fused K norm+RoPE) while using
the correct AITER quantization for K-cache storage.

Note: _store_index_k_cache already has this _is_cuda guard for the same reason.
"""
import sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"

with open(FILE, "r") as f:
    content = f.read()

old = """        if (
            not _is_fp8_fnuz
            and out_cache_loc is not None
            and can_use_dsa_fused_store(torch.bfloat16, out_cache_loc.dtype, page_size)
        ):
            fused_k_indexer_norm_rope_store("""

new = """        if (
            _is_cuda
            and not _is_fp8_fnuz
            and out_cache_loc is not None
            and can_use_dsa_fused_store(torch.bfloat16, out_cache_loc.dtype, page_size)
        ):
            fused_k_indexer_norm_rope_store("""

if old in content:
    content = content.replace(old, new, 1)
    with open(FILE, "w") as f:
        f.write(content)
    print("[OK] Added _is_cuda guard to _fused_k_prepare_and_store fused store path")
elif "_is_cuda\n            and not _is_fp8_fnuz\n            and out_cache_loc is not None\n            and can_use_dsa_fused_store" in content:
    print("[SKIP] _is_cuda guard already present")
else:
    print("[ERROR] Pattern not found in _fused_k_prepare_and_store")
    sys.exit(1)
