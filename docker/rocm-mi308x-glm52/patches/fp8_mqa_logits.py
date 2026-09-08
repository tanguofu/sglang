# Patched AITER fp8_mqa_logits for MI308X (gfx942)
# Original: BLOCK_KV=128, num_stages=2 → 80KB shared memory (exceeds 64KB limit)
# Patched:  BLOCK_KV=64,  num_stages=1 → fits in 64KB shared memory
#
# This file replaces /sgl-workspace/aiter/aiter/ops/triton/attention/fp8_mqa_logits.py
# in the Docker image to fix OutOfResources on gfx942.

import torch

from aiter.ops.triton._triton_kernels.attention.fp8_mqa_logits import (
    _fp8_mqa_logits_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
import inspect
from packaging.version import Version
import triton

TRITON_VERSION = Version(triton.__version__)
TRITON_GE_36 = TRITON_VERSION >= Version("3.6.0")

arch = arch_info.get_arch()
_gluon_fp8_mqa_logits_kernel = None
if TRITON_GE_36:
    try:
        if arch == "gfx950":
            from aiter.ops.triton._gluon_kernels.gfx950.attention.fp8_mqa_logits import (
                _gluon_fp8_mqa_logits_kernel,
            )
        elif arch == "gfx1250":
            from aiter.ops.triton._gluon_kernels.gfx1250.attention.fp8_mqa_logits import (
                _gluon_fp8_mqa_logits_kernel,
            )
    except Exception:
        _gluon_fp8_mqa_logits_kernel = None


def _async_copy_accepts_distributed_layout() -> bool:
    try:
        from triton.experimental.gluon.language.amd.cdna4 import async_copy
        src = inspect.getsource(async_copy.global_load_to_shared)
    except (OSError, TypeError, ImportError, AttributeError):
        return False
    return "DistributedLayout" in src


def _permute_accepts_constexpr_tuple() -> bool:
    try:
        from triton.language.core import _unwrap_iterable, constexpr
    except ImportError:
        return False
    probe = constexpr((0, 1, 2))
    result = _unwrap_iterable((probe,))
    return not isinstance(result, constexpr)


ASYNC_COPY_SUPPORTS_DISTRIBUTED = _async_copy_accepts_distributed_layout()
FOLDED_REDUCTED_SUPPORT = _permute_accepts_constexpr_tuple()


def fp8_mqa_logits(
    Q,
    KV,
    kv_scales,
    weights,
    cu_starts,
    cu_ends,
    clean_logits=True,
):
    """
    Patched for MI308X (gfx942): BLOCK_KV=64, num_stages=1 to fit in 64KB shared memory.
    """
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."
    aligned_size = 256
    seq_len_kv_aligned = (seq_len_kv + aligned_size - 1) // aligned_size * aligned_size
    if clean_logits:
        logits = torch.full(
            (seq_len, seq_len_kv_aligned),
            fill_value=-float("inf"),
            dtype=torch.float32,
            device=Q.device,
        )[:, :seq_len_kv]
    else:
        logits = torch.empty(
            (seq_len, seq_len_kv_aligned),
            dtype=torch.float32,
            device=Q.device,
        )[:, :seq_len_kv]

    use_gluon = TRITON_GE_36 and _gluon_fp8_mqa_logits_kernel is not None
    stride_q_s, stride_q_h, stride_q_d = Q.stride()
    stride_kv_s, stride_kv_d = KV.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()
    if not use_gluon:
        # PATCH: BLOCK_KV=64 (was 128), num_stages=1 (was 2) for gfx942 shared memory
        block_kv = 64

        matrix_instr_nonkdim = 32
        if seq_len <= 1024:
            matrix_instr_nonkdim = 16

        _fp8_mqa_logits_kernel[(seq_len,)](
            Q_ptr=Q,
            KV_ptr=KV,
            kv_scales_ptr=kv_scales,
            weights_ptr=weights,
            cu_start_ptr=cu_starts,
            cu_end_ptr=cu_ends,
            logits_ptr=logits,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            NUM_HEADS=num_heads,
            HEAD_SIZE=head_size,
            stride_q_s=stride_q_s,
            stride_q_h=stride_q_h,
            stride_q_d=stride_q_d,
            stride_kv_s=stride_kv_s,
            stride_kv_d=stride_kv_d,
            stride_w_s=stride_w_s,
            stride_w_h=stride_w_h,
            stride_logits_s=stride_logits_s,
            stride_logits_k=stride_logits_k,
            BLOCK_KV=block_kv,
            num_warps=4,
            num_stages=1,  # PATCH: was 2
            waves_per_eu=2,
            matrix_instr_nonkdim=matrix_instr_nonkdim,
        )
    else:
        num_buffers = 2
        USE_FOLDED_REDUCTION = FOLDED_REDUCTED_SUPPORT and num_heads > 16
        if arch == "gfx950":
            block_kv = 128
            num_warps = 4
            waves_per_eu = 2
            use_buffer_load = True
            use_buffer_store = True
        else:
            block_kv = 128
            num_warps = 4
            waves_per_eu = 2
            use_buffer_load = ASYNC_COPY_SUPPORTS_DISTRIBUTED
            use_buffer_store = ASYNC_COPY_SUPPORTS_DISTRIBUTED

        other = {}
        if USE_FOLDED_REDUCTION:
            other["USE_FOLDED_REDUCTION"] = True

        _gluon_fp8_mqa_logits_kernel[(seq_len,)](
            Q_ptr=Q,
            KV_ptr=KV,
            kv_scales_ptr=kv_scales,
            weights_ptr=weights,
            cu_start_ptr=cu_starts,
            cu_end_ptr=cu_ends,
            logits_ptr=logits,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            NUM_HEADS=num_heads,
            HEAD_SIZE=head_size,
            stride_q_s=stride_q_s,
            stride_q_h=stride_q_h,
            stride_q_d=stride_q_d,
            stride_kv_s=stride_kv_s,
            stride_kv_d=stride_kv_d,
            stride_w_s=stride_w_s,
            stride_w_h=stride_w_h,
            stride_logits_s=stride_logits_s,
            stride_logits_k=stride_logits_k,
            BLOCK_KV=block_kv,
            NUM_WARPS=num_warps,
            NUM_BUFFERS=num_buffers,
            NUM_CHAINS=num_buffers,
            USE_BUFFER_LOAD=use_buffer_load,
            USE_BUFFER_STORE=use_buffer_store,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            **other,
        )

    return logits
