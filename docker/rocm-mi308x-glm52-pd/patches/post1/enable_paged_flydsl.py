#!/usr/bin/env python3
"""Enable FlyDSL mfma_r4_w4 for the DSA paged/decode MQA logits path.

The v0.5.17 image already uses FlyDSL for ragged/prefill MQA logits. This patch
adds an env-gated decode path that gathers paged index-K into the same ragged
layout and reuses the verified FlyDSL mfma_r4_w4 kernel. It remains opt-in via
SGLANG_DSA_PAGED_FLYDSL=1 so the AITER paged kernel is the rollback path.
"""
import sys
from pathlib import Path


path = Path(sys.argv[1])
src = path.read_text()

branch_anchor = "        if self.paged_mqa_logits_backend.is_aiter():\n"
branch_replacement = '''        row_starts = None
        if _is_hip and get_bool_env_var("SGLANG_DSA_PAGED_FLYDSL"):
            from aiter.ops.flydsl.kernels.fp8_mqa_logits import (
                flydsl_fp8_mqa_logits,
            )

            indexer_seq_lens = metadata.get_indexer_seq_len()
            indexer_seq_lens_cpu = metadata.get_indexer_seq_len_cpu()
            seq_len_sum = int(indexer_seq_lens_cpu.sum().item())
            max_indexer_seq_len = int(indexer_seq_lens_cpu.max().item())
            k_fp8, k_scale = get_token_to_kv_pool().get_index_k_scale_buffer(
                layer_id,
                indexer_seq_lens,
                block_tables,
                seq_len_sum,
                max_indexer_seq_len,
            )
            if _is_fp8_fnuz:
                k_fp8 = k_fp8.view(torch.float8_e4m3fnuz)
            else:
                k_fp8 = k_fp8.view(torch.float8_e4m3fn)
            k_scale = k_scale.view(torch.float32).squeeze(-1)
            ks, ke = metadata.get_indexer_kvcache_range()
            logits = flydsl_fp8_mqa_logits(
                q_fp8[:q_offset],
                k_fp8,
                k_scale,
                weights[:q_offset],
                ks,
                ke,
                clean_logits=False,
                variant="mfma_r4_w4",
            )
            row_starts = ks
        elif self.paged_mqa_logits_backend.is_aiter():
'''

if "SGLANG_DSA_PAGED_FLYDSL" not in src:
    assert branch_anchor in src, "paged FlyDSL: AITER branch anchor missing"
    src = src.replace(branch_anchor, branch_replacement, 1)

mask_old = '''        self._mask_init_and_local_tokens(logits, seqlens_32)
        topk_result = metadata.topk_transform(logits[:q_offset], self.index_topk)
'''
mask_new = '''        self._mask_init_and_local_tokens(logits, seqlens_32, row_starts)
        topk_result = metadata.topk_transform(
            logits[:q_offset], self.index_topk, ks=row_starts
        )
'''
if "ks=row_starts" not in src:
    assert mask_old in src, "paged FlyDSL: top-k tail anchor missing"
    src = src.replace(mask_old, mask_new, 1)

path.write_text(src)
print(f"[ok] {path}: paged FlyDSL mfma_r4_w4 enabled")
