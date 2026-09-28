from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)

from .utils import make_name


@cache_once
def _jit_topk_v1_module():
    # topk (<= 1024) is a runtime argument, not a compile-time constant, so a
    # single module serves every k. Baking it in via -DSGL_TOPK used to build one
    # module per k, and since the macro fed a `constexpr` rather than a template
    # parameter every module exported identically mangled symbols -- see the
    # comment in topk_v1.cuh for how that broke the second module's launch.
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk_v1"),
        *args,
        cuda_files=["deepseek_v4/topk_v1.cuh"],
        cuda_wrappers=[("topk_transform", f"TopKKernel<{args}>::transform")],
    )


@cache_once
def _jit_topk_v2_module():
    # v2 is universal: topk (<= 2048) is a runtime argument, not a compile-time
    # constant, so a single module serves every k. The upstream split of the
    # single `transform` entry into paged/ragged/packed (PR #37889) renamed the
    # wrappers; the packed one is compiled under USE_ROCM only.
    args = make_cpp_args(is_arch_support_pdl())
    kernel = f"TopKKernel<{args}>"
    wrappers = [
        ("topk_transform_paged", f"{kernel}::transform_paged"),
        ("topk_transform_ragged", f"{kernel}::transform_ragged"),
        ("topk_plan", f"{kernel}::plan"),
    ]
    if is_hip_runtime():
        wrappers.append(("topk_transform_packed", f"{kernel}::transform_packed"))
    return load_jit(
        make_name("topk_v2"),
        cuda_files=["deepseek_v4/topk_v2.cuh"],
        cuda_wrappers=wrappers,
    )


def topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    if is_hip_runtime():
        torch.ops.sgl_kernel.deepseek_v4_topk_transform_512(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )
    else:
        module = _jit_topk_v1_module()
        module.topk_transform(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )


# metadata is (batch+1, 2) int32: row 0 = {cluster_threshold, num_cluster_items};
# rows 1..N = {batch_id, seq_len} of items routed to the persistent cluster pool.
_PLAN_METADATA_INTS_PER_BATCH = 2


def plan_topk_v2(
    seq_lens: torch.Tensor,
    static_threshold: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Preprocess the per-batch routing plan for :func:`topk_transform_512_v2`.

    IMPORTANT: every entry of ``seq_lens`` must be NON-NEGATIVE. The device
    kernel reads the int32 buffer as ``uint32_t``, so a negative length (e.g.
    -4 from a DP-padded / idle-companion row) reinterprets as ~4e9, poisons
    the plan, and drives the transform kernel into an illegal memory access.
    Producers of padded rows must clamp their lengths to 0 (0 selects the
    trivial all-(-1) output path, which is safe).

    ``out``, when given, must be a plan previously produced by this function
    for the same batch size (shape ``(bs + 1, _PLAN_METADATA_INTS_PER_BATCH)``)
    and is rewritten in place instead of allocating a fresh buffer.
    """
    module = _jit_topk_v2_module()
    if out is None:
        bs = seq_lens.shape[0]
        out = seq_lens.new_empty(bs + 1, _PLAN_METADATA_INTS_PER_BATCH)
    module.topk_plan(seq_lens, out, static_threshold)
    return out


def topk_transform_ragged_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    out_offsets: torch.Tensor,
    out_indices: torch.Tensor,
    row_starts: Optional[torch.Tensor] = None,
) -> None:
    """Ragged (prefill) fused top-k for a contiguous-KV score matrix.

    Row ``i`` selects the top-k of ``scores[i, ks : ks + seq_lens[i]]`` (``ks =
    row_starts[i]``, 0 when ``row_starts`` is omitted) and writes
    ``selected_position + out_offsets[i]`` into ``out_indices``, ``-1`` padded.
    With the production convention ``out_offsets == row_starts`` that is the
    column index itself, i.e. the token's slot in the batch's flattened KV.

    Unlike :func:`topk_transform_512_v2` this needs no page table and no plan
    (the cluster path only pays off for very few rows, and prefill has many).

    IMPORTANT: ``scores`` is written in place -- the <= 3 columns ahead of each
    row's window that the 16-byte-aligned read base pulls in are masked out.
    They are invalid for that row and the buffer must have no other consumer.
    ``seq_lens`` entries must be NON-NEGATIVE, as for the paged entry point.
    """
    module = _jit_topk_v2_module()
    module.topk_transform_ragged(scores, seq_lens, row_starts, out_offsets, out_indices)


def topk_transform_512_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: Optional[torch.Tensor],
    out_page_indices: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor,
) -> None:
    """Fused top-k + optional page-table transform (DeepSeek-V4 top-k v2 kernel).

    Two output modes, chosen by whether ``page_tables`` is given and resolved to
    a device-side template parameter, so an unused page-table gather is compiled
    out rather than skipped at runtime:

    * ``page_tables=None`` -- ``out_page_indices`` receives the raw selected
      indices and no page table is read.
    * ``page_tables`` given -- ``out_page_indices`` receives the page-table
      transform of them.

    IMPORTANT: every entry of ``seq_lens`` must be NON-NEGATIVE, and
    ``metadata`` must come from :func:`plan_topk_v2` over the same ``seq_lens``
    values. The kernel reads lengths as ``uint32_t``: a negative entry
    reinterprets as a ~4e9-token sequence, sending the row down the cluster
    path over garbage scores and crashing with an illegal memory access
    (GLM 5.2 MTP DP-idle companion rows hit exactly this). A length of 0 is
    the valid way to express "no tokens": the row takes the trivial path and
    the output is all -1.
    """
    module = _jit_topk_v2_module()
    module.topk_transform_paged(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        metadata,
    )


def topk_transform_packed_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    *,
    row_starts: torch.Tensor,
    row_to_batch: Optional[torch.Tensor] = None,
) -> None:
    """Packed (DSA extend prefill) fused top-k + page-table transform.

    Row ``i`` selects the top-k of ``scores[i, ks : ks + seq_lens[i]]``
    (``ks = row_starts[i]``) and writes the page-table transform of the selected
    row-local positions into ``out_page_indices``, ``-1`` padded. Prefill expands
    one request into many query-token rows, so ``row_to_batch[i]`` (optional,
    ``(rows,)`` int32) names the ``page_tables`` row of the request row ``i``
    belongs to; omitting it indexes the table by score row. ``row_to_batch`` is
    not range-checked.

    Ported from PR #37889 (amd/topk-v2-prefill). No plan and no cluster path:
    the implementation level is picked per row at runtime.

    NOTE: ``scores`` is MODIFIED IN PLACE -- the <= 3 columns ahead of each
    row's window that the 16-byte-aligned read base pulls in are masked out.
    They are invalid for that row and the buffer must have no other consumer,
    so do not pass a view with overlapping rows.
    ``seq_lens`` entries must be NON-NEGATIVE, as for the paged entry point.

    ROCm only: the kernel is compiled under ``USE_ROCM`` so that CUDA and XPU
    builds are untouched.
    """
    assert is_hip_runtime(), "topk_transform_packed_v2 is compiled under USE_ROCM only"
    module = _jit_topk_v2_module()
    module.topk_transform_packed(
        scores,
        seq_lens,
        row_starts,
        page_tables,
        out_page_indices,
        page_size,
        row_to_batch,
    )
