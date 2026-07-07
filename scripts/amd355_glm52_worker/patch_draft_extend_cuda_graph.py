#!/usr/bin/env python3
"""Patch to enable draft extend CUDA graph for DSA backend on HIP.

On HIP, DeepseekSparseAttnBackend is only added to graph_supported_backend_types
for CUDA/MUSA, not HIP. This patch extends it to HIP as well, and enables
supports_cuda_draft_extend_graph for HIP.
"""
import os

SGLANG_DIR = "/sgl-workspace/sglang/python/sglang/srt"
path = os.path.join(SGLANG_DIR, "speculative/eagle_worker_v2.py")
with open(path, "r") as f:
    content = f.read()

# Patch 1: Add DSA to graph_supported_backend_types for HIP
old1 = """        if _is_cuda or _is_musa:
            # DSA is CUDA-only; import lazily so non-CUDA builds don't pull in
            # deep_gemm and the rest of the sparse-attention stack at import time.
            from sglang.srt.layers.attention.dsa_backend import (
                DeepseekSparseAttnBackend,
            )

            graph_supported_backend_types.append(DeepseekSparseAttnBackend)"""

new1 = """        if _is_cuda or _is_musa or _is_hip:
            # DSA is CUDA-only; import lazily so non-CUDA builds don't pull in
            # deep_gemm and the rest of the sparse-attention stack at import time.
            from sglang.srt.layers.attention.dsa_backend import (
                DeepseekSparseAttnBackend,
            )

            graph_supported_backend_types.append(DeepseekSparseAttnBackend)"""

if old1 in content:
    content = content.replace(old1, new1)
    print("[OK] Patched graph_supported_backend_types to include HIP")
else:
    print("[SKIP] graph_supported_backend_types pattern not found")

# Patch 2: Enable supports_cuda_draft_extend_graph for HIP
old2 = """        supports_cuda_draft_extend_graph = (
            _is_cuda or _is_musa
        ) and graph_supported_backend"""

new2 = """        supports_cuda_draft_extend_graph = (
            _is_cuda or _is_musa or _is_hip
        ) and graph_supported_backend"""

if old2 in content:
    content = content.replace(old2, new2)
    print("[OK] Patched supports_cuda_draft_extend_graph to include HIP")
else:
    print("[SKIP] supports_cuda_draft_extend_graph pattern not found")

with open(path, "w") as f:
    f.write(content)
print("\nDone. Restart SGLang to apply.")
