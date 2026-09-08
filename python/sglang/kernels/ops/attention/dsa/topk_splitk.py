"""SplitK top-k transform for DSA decode (gfx942) — torch.topk fast path.

Profile (2026-09-02, decode-0 pod, MI308X): the AOT
topk_transform_decode_kernel launches one 1024-thread block per row —
4 blocks on a 220-CU GPU at MTP verify — and each block serially
radix-scans the full logits row. Measured vs a torch.topk + gather path
(both numerically identical, sets_equal on all shapes):

    B=4 seq= 64K: aot  52.1µs  torch  78.8µs  (0.7x — AOT wins)
    B=4 seq=128K: aot  86.6µs  torch 118.2µs  (0.7x — AOT wins)
    B=4 seq=256K: aot 152.5µs  torch 111.4µs  (1.4x — torch wins)
    B=4 seq=512K: aot 242.7µs  torch 154.1µs  (1.6x — torch wins)
    B=4 seq=1024K: aot 623.6µs torch 216.3µs  (2.9x — torch wins)
    B=8 seq=256K: aot 152.9µs  torch 165.9µs  (0.9x — AOT wins)

torch.topk (CUB device-wide sort) scales with total elements; the AOT
kernel's per-row serial scan does not. Dispatch on (rows, max_len):
use this path only when rows <= 4 and max_len >= 256K — the MTP-verify
long-context case. Everything else keeps the AOT kernel.

Gated by SGLANG_DSA_TOPK_SPLITK=1 (default off).
"""

from __future__ import annotations

import torch

# Dispatch thresholds from the measured crossover points above.
_MIN_ROWS_FOR_SPLITK = 4  # rows <= this → consider splitk
_MIN_LEN_FOR_SPLITK = 256 * 1024  # max row length >= this → splitk


def should_use_splitk_topk(rows: int, max_len: int) -> bool:
    """torch.topk path wins only for few-row, long-context shapes."""
    return rows <= _MIN_ROWS_FOR_SPLITK and max_len >= _MIN_LEN_FOR_SPLITK


def splitk_topk_transform_decode(
    logits: torch.Tensor,  # [B, stride] fp32
    lengths: torch.Tensor,  # [B] i32
    page_table_1: torch.Tensor,  # [B, stride] i32, one entry per token
    topk: int,
) -> torch.Tensor:
    """torch.topk + gather; drop-in for fast_topk_transform_fused decode path.

    Returns [B, topk] page-table entries (same contract as the AOT kernel).
    """
    _, idx = torch.topk(logits, topk, dim=-1)  # [B, topk] token indices
    return torch.gather(page_table_1, 1, idx)
