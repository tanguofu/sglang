#!/usr/bin/env python3
"""Port patches 1.6 + 04 + fused-store-guard + i32-overflow to post1 dsa_indexer.py.

Patch 1.6 (k_norm fp32 cast): the fused_k_indexer_norm_rope{,_store} calls pass
k_norm weight/bias that may be low-precision on HIP. Add two helper properties that
force-cast to float32, and use them at the call sites.

Patch 04 (breakable-target-verify): in piecewise/breakable CUDA graph mode, `metadata`
is set to None before the decode/idle/target_verify/draft_extend_v2 branch, then used
directly in _get_topk_paged -> crash. Guard: if metadata is None, re-fetch via
get_attn_backend().get_indexer_metadata(); if still None, return None.

Patch fused-store-guard: the combined fused_k_indexer_norm_rope_store kernel has no
_is_cuda guard and runs on HIP, corrupting index K cache at long context (>4096 tokens).
Guard: only use fused-store when max seq_len <= 4096; fall back to safe separate K
kernel + store for longer sequences. Proven fix from iwiki doc 4024854008.

Patch i32-overflow: guard against i32 index overflow in FlyDSL/Triton MQA logits
kernel at long context (k_offset > ~180k wraps i32). Idempotent — skipped if the
base already has it.

Idempotent.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/port-post1/base/srt_layers_attention_dsa_dsa_indexer.py")
src = path.read_text()
changed = []


def apply(label, did):
    changed.append(f"{label}: {'applied' if did else 'skipped'}")


# ---------- Patch 1.6 ----------
if "_k_norm_weight_f32" in src:
    apply("1.6 property", False)
else:
    # Insert the two properties just before `_fused_k_weights` (present in both
    # branch and base). Find the def line.
    anchor = "    def _fused_k_weights(self"
    assert anchor in src, "1.6: anchor _fused_k_weights not found"
    props = (
        "    @property\n"
        "    def _k_norm_weight_f32(self) -> torch.Tensor:\n"
        "        w = self.k_norm.weight\n"
        "        return w.float() if w.dtype != torch.float32 else w\n"
        "\n"
        "    @property\n"
        "    def _k_norm_bias_f32(self) -> torch.Tensor:\n"
        "        b = self.k_norm.bias\n"
        "        return b.float() if b is not None and b.dtype != torch.float32 else b\n"
        "\n"
    )
    src = src.replace(anchor, props + anchor, 1)
    apply("1.6 property", True)

# Replace the two call sites: self.k_norm.weight -> self._k_norm_weight_f32,
# self.k_norm.bias -> self._k_norm_bias_f32, ONLY inside the fused_k_indexer calls.
# The k_norm.weight/bias appear in exactly the store + fallback call sites.
# Do targeted multi-line replacements to be safe.
# Two call sites with different indentation (16-space inside fused_store, 12-space
# inside fallback). Replace self.k_norm.weight/bias -> _f32 variants at both.
n_weight = src.count("self.k_norm.weight,")
n_bias = src.count("self.k_norm.bias,")
if "_k_norm_weight_f32,\n" in src and n_weight == 0:
    apply("1.6 call sites", False)
else:
    assert n_weight == 2, f"1.6: expected 2 k_norm.weight call sites, found {n_weight}"
    assert n_bias == 2, f"1.6: expected 2 k_norm.bias call sites, found {n_bias}"
    src = src.replace("self.k_norm.weight,", "self._k_norm_weight_f32,")
    src = src.replace("self.k_norm.bias,", "self._k_norm_bias_f32,")
    apply("1.6 call sites", True)

# ---------- Patch 04 ----------
# v0.5.17 already includes the metadata None guard (3 occurrences of
# "metadata is None").  Skip gracefully if the old anchor is not found.
marker = "# FIX(breakable-target-verify): metadata None guard"
if marker in src:
    apply("04", False)
else:
    # Locate the decode/idle/target_verify branch and insert the guard right after it.
    branch_old = (
        "            if (\n"
        "                forward_batch.forward_mode.is_decode_or_idle()\n"
        "                or forward_batch.forward_mode.is_target_verify()\n"
        "                or forward_batch.forward_mode.is_draft_extend_v2()\n"
        "            ):\n"
        "                topk_result = self._get_topk_paged(\n"
    )
    branch_new = (
        "            if (\n"
        "                forward_batch.forward_mode.is_decode_or_idle()\n"
        "                or forward_batch.forward_mode.is_target_verify()\n"
        "                or forward_batch.forward_mode.is_draft_extend_v2()\n"
        "            ):\n"
        "                # FIX(breakable-target-verify): metadata None guard\n"
        "                if metadata is None:\n"
        "                    metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)\n"
        "                    if metadata is None:\n"
        "                        return None\n"
        "                topk_result = self._get_topk_paged(\n"
    )
    if branch_old in src:
        src = src.replace(branch_old, branch_new, 1)
        apply("04", True)
    else:
        # v0.5.17 already has this fix — skip gracefully
        apply("04 (already in base)", False)

# ---------- Patch fused-store-length-guard ----------
guard_marker = "# FIX(fused-store-length-guard): long-context safety"
if guard_marker in src:
    apply("fused-store-guard", False)
else:
    # The _fused_k_prepare_and_store method calls fused_k_indexer_norm_rope_store
    # without an _is_cuda guard. On HIP this corrupts KV cache at long context.
    # Add a max seq_len guard (<=4096) before the can_use_dsa_fused_store check.
    old_guard = (
        "        if (\n"
        "            not _is_fp8_fnuz\n"
        "            and out_cache_loc is not None\n"
        "            and can_use_dsa_fused_store(torch.bfloat16, out_cache_loc.dtype, page_size)\n"
        "        ):"
    )
    new_guard = (
        "        # FIX(fused-store-length-guard): long-context safety on HIP\n"
        "        _max_ctx = 0\n"
        "        if forward_batch.seq_lens_cpu is not None and len(forward_batch.seq_lens_cpu) > 0:\n"
        "            _max_ctx = int(forward_batch.seq_lens_cpu.max().item())\n"
        "        if (\n"
        "            not _is_fp8_fnuz\n"
        "            and out_cache_loc is not None\n"
        "            and _max_ctx <= 4096\n"
        "            and can_use_dsa_fused_store(torch.bfloat16, out_cache_loc.dtype, page_size)\n"
        "        ):"
    )
    assert old_guard in src, "fused-store-guard: anchor not found in _fused_k_prepare_and_store"
    src = src.replace(old_guard, new_guard, 1)
    apply("fused-store-guard", True)

# ---------- Patch i32-overflow (idempotent, skipped if base already has it) ----------
if "i32_safe_rows" in src:
    apply("i32-overflow", False)
else:
    # Guard against i32 index overflow in FlyDSL/Triton MQA logits kernel.
    # The kernel indexes logits with i32, so long-context prefill
    # (k_offset > ~180k) wraps the i32 index and corrupts output.
    # Insert before the cu_seqlens_q_full allocation in the MQA logits path.
    # NOTE: v0.5.17 uses 12-space indentation for this block (inside a with/if).
    i32_anchor = "            actual_seq_q = torch.tensor([actual_seq_q], dtype=torch.int32).to("
    if i32_anchor in src:
        i32_fix = (
            "            # Guard against i32 index overflow in the FlyDSL/Triton MQA logits\n"
            "            # kernel: the kernel indexes logits with i32, so long-context\n"
            "            # prefill (k_offset > ~180k) wraps the i32 index and corrupts output.\n"
            "            INT32_MAX = 2147483647\n"
            "            bytes_per_row = self.index_topk * 4  # fp32 logits per row\n"
            "            i32_safe_rows = INT32_MAX // max(bytes_per_row, 1)\n"
            "            max_rows = min(max_rows, i32_safe_rows, q_offset)\n"
            "\n"
        )
        src = src.replace(i32_anchor, i32_fix + i32_anchor, 1)
        apply("i32-overflow", True)
    else:
        # Anchor not found — may be a different base version. Skip gracefully.
        apply("i32-overflow (anchor not found, skipped)")

path.write_text(src)
print(f"[ok] {path}: " + ", ".join(changed))
