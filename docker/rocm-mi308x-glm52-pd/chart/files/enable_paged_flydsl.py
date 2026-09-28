#!/usr/bin/env python3
"""Enable a native paged FlyDSL MFMA kernel for DSA decode MQA logits."""
import sys
from pathlib import Path


path = Path(sys.argv[1])
src = path.read_text()

branch_anchor = "        if self.paged_mqa_logits_backend.is_aiter():\n"
aiter_elif_anchor = "        elif self.paged_mqa_logits_backend.is_aiter():\n"
old_branch_start = "        if _is_hip and get_bool_env_var(\"SGLANG_DSA_PAGED_FLYDSL\"):\n"
old_branch_import = "                flydsl_fp8_mqa_logits,\n"
branch_replacement = '''        if (
            _is_hip
            and get_bool_env_var("SGLANG_DSA_PAGED_FLYDSL")
            and (
                (page_size == 1 and not _use_aiter_preshuffle)
                or (_use_aiter_preshuffle and page_size % 16 == 0)
            )
        ):
            from aiter.ops.flydsl.kernels.fp8_mqa_logits import (
                flydsl_fp8_paged_mqa_logits,
            )

            if next_n == 4:
                paged_flydsl_variant = "mfma_r4_w4"
            elif next_n == 2:
                paged_flydsl_variant = "mfma_r2_w4"
            else:
                paged_flydsl_variant = "mfma_r1_w4"

            num_blocks = kv_cache_fp8.shape[0]
            kv_words = kv_cache_fp8.reshape(num_blocks, -1).view(torch.float32)
            kv_scales = kv_words[:, page_size * 32:]
            kv_flat = kv_cache_fp8.reshape(-1)
            if _is_fp8_fnuz:
                kv_flat = kv_flat.view(torch.float8_e4m3fnuz)
            else:
                kv_flat = kv_flat.view(torch.float8_e4m3fn)
            seqlens_for_flydsl = (
                seqlens_32.reshape(-1) if seqlens_32.dim() == 2 else seqlens_32
            )
            rows = q_fp8[:q_offset].shape[0]
            tables = block_tables
            if tables.shape[0] != rows:
                if tables.shape[0] == 0 or rows % tables.shape[0] != 0:
                    raise ValueError(
                        f"paged FlyDSL block table rows mismatch: tables={tables.shape[0]}, q_rows={rows}"
                    )
                tables = tables.repeat_interleave(rows // tables.shape[0], dim=0)
            if (
                seqlens_for_flydsl.shape[0] != rows
            ):
                if seqlens_for_flydsl.shape[0] == 0 or rows % seqlens_for_flydsl.shape[0] != 0:
                    raise ValueError(
                        f"paged FlyDSL context length rows mismatch: context_lens={seqlens_for_flydsl.shape[0]}, q_rows={rows}"
                    )
                seqlens_for_flydsl = seqlens_for_flydsl.repeat_interleave(
                    rows // seqlens_for_flydsl.shape[0]
                )
            logits = flydsl_fp8_paged_mqa_logits(
                q_fp8[:q_offset],
                kv_flat,
                kv_scales,
                weights[:q_offset],
                seqlens_for_flydsl,
                tables,
                max_seq_len,
                variant=paged_flydsl_variant,
                page_size=page_size,
                preshuffle=_use_aiter_preshuffle,
            )
            row_starts = None
        elif self.paged_mqa_logits_backend.is_aiter():
'''

if "flydsl_fp8_paged_mqa_logits" not in src:
    old_start = src.find(old_branch_start)
    if old_start >= 0 and old_branch_import in src[old_start : old_start + 400]:
        old_end = src.find(aiter_elif_anchor, old_start)
        if old_end < 0:
            raise SystemExit("paged FlyDSL: AITER elif after old ragged branch missing")
        src = src[:old_start] + branch_replacement + src[old_end + len(aiter_elif_anchor) :]
    else:
        if branch_anchor not in src:
            raise SystemExit("paged FlyDSL: AITER branch anchor missing")
        src = src.replace(branch_anchor, branch_replacement, 1)

path.write_text(src)
print(f"[ok] {path}: native paged FlyDSL MFMA enabled")
