#!/usr/bin/env python3
"""Apply DP8/EP8 + MTP correctness fixes to the v0.5.17 image tree.

The image intentionally applies semantic patches to its vendored source instead of
copying the diverged local Python tree. This script ports two fixes required by
DP-attention with speculative decoding:

* dsa_backend.py: expand idle speculative metadata to one row per query row.
* dsa_indexer.py: exclude HIP padding rows from paged MQA logits and top-k.

The script is idempotent and fails when an expected base anchor is missing.
"""
import sys
from pathlib import Path


ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python/sglang")
BACKEND = ROOT / "srt/layers/attention/dsa_backend.py"
INDEXER = ROOT / "srt/layers/attention/dsa/dsa_indexer.py"


def patch_backend() -> str:
    src = BACKEND.read_text()
    changed = False

    helper_anchor = "# Reuse this workspace buffer across all DSA backend instances"
    helper = '''def _idle_spec_rows_per_seq(forward_batch: ForwardBatch) -> int:
    """Query rows carried by each padded sequence of an idle DP-attention rank."""
    if not forward_batch.forward_mode.is_idle():
        return 1
    spec_info = forward_batch.spec_info
    if spec_info is None:
        return 1
    return max(1, spec_info.num_tokens_per_req)


'''
    if "_idle_spec_rows_per_seq" not in src:
        assert helper_anchor in src, "dp8ep8_mtp_fix: dsa_backend helper anchor missing"
        src = src.replace(helper_anchor, helper + helper_anchor, 1)
        changed = True

    decode_anchor = '''        if forward_batch.forward_mode.is_decode_or_idle():
            extend_seq_lens_cpu = [1] * batch_size
'''
    decode_replacement = '''        if forward_batch.forward_mode.is_decode_or_idle():
            rows_per_seq = _idle_spec_rows_per_seq(forward_batch)
            if rows_per_seq > 1:
                batch_size = batch_size * rows_per_seq
                cache_seqlens_int32 = cache_seqlens_int32.repeat_interleave(
                    rows_per_seq
                )
                cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
                page_table = page_table.repeat_interleave(rows_per_seq, dim=0)
                indexer_seq_lens = indexer_seq_lens.repeat_interleave(rows_per_seq)
                if indexer_seq_lens_cpu is not None:
                    indexer_seq_lens_cpu = indexer_seq_lens_cpu.repeat_interleave(
                        rows_per_seq
                    )
            extend_seq_lens_cpu = [1] * batch_size
'''
    if "rows_per_seq = _idle_spec_rows_per_seq(forward_batch)" not in src:
        assert decode_anchor in src, "dp8ep8_mtp_fix: dsa_backend decode anchor missing"
        src = src.replace(decode_anchor, decode_replacement, 1)
        changed = True

    if changed:
        BACKEND.write_text(src)
    return "applied" if changed else "already-present"


def patch_indexer() -> str:
    src = INDEXER.read_text()
    changed = False

    replacements = [
        (
            "            deepgemm_fp8_paged_mqa_logits(\n"
            "                q_fp8,\n"
            "                kv_cache_fp8,\n"
            "                weights,\n",
            "            deepgemm_fp8_paged_mqa_logits(\n"
            "                q_fp8[:q_offset],\n"
            "                kv_cache_fp8,\n"
            "                weights[:q_offset],\n",
        ),
        (
            "        topk_result = metadata.topk_transform(logits, self.index_topk)",
            "        topk_result = metadata.topk_transform(\n"
            "            logits[:q_offset], self.index_topk\n"
            "        )",
        ),
        (
            "        if not _is_hip and q_offset < q_fp8.shape[0]:",
            "        if q_offset < q_fp8.shape[0]:",
        ),
    ]
    for old, new in replacements:
        if new in src:
            continue
        assert old in src, f"dp8ep8_mtp_fix: dsa_indexer anchor missing: {old[:60]!r}"
        src = src.replace(old, new, 1)
        changed = True

    if changed:
        INDEXER.write_text(src)
    return "applied" if changed else "already-present"


backend_status = patch_backend()
indexer_status = patch_indexer()
print(f"[ok] dp8ep8_mtp_fix: backend={backend_status}, indexer={indexer_status}")
