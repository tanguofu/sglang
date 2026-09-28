"""Eligibility for gfx942 DSA indexer projection fusion.

CUDA keeps full indexer fusion (projection + rotary/quant/cache). HIP gfx942
can fuse only the paired ``wk`` / ``weights_proj`` GEMM behind
``SGLANG_ROCM_DSA_INDEXER_PROJECTION_FUSION`` and still use the legacy
Hadamard / cache path. See sgl-project/sglang#39243.
"""

from __future__ import annotations

from typing import Any, Optional


def dsa_indexer_projection_fusion_enabled(
    *,
    cuda_full_fusion: bool,
    is_hip: bool,
    gfx942: bool,
    enable_rocm_proj: bool,
    disable_fusion: bool,
    is_neox_style: bool,
    enable_lora: bool,
    quant_config: Optional[Any],
) -> bool:
    if cuda_full_fusion:
        return True
    if not (
        is_hip
        and gfx942
        and enable_rocm_proj
        and not disable_fusion
        and not is_neox_style
        and not enable_lora
    ):
        return False
    if quant_config is None:
        return True
    block = getattr(quant_config, "weight_block_size", None)
    return (
        quant_config.get_name() == "fp8"
        and not getattr(quant_config, "use_mxfp8", False)
        and list(block or []) == [128, 128]
    )
