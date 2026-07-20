#!/usr/bin/env python3
"""Port patches 1.6 + 04 to post1 dsa_indexer.py.

Patch 1.6 (k_norm fp32 cast): the fused_k_indexer_norm_rope{,_store} calls pass
k_norm weight/bias that may be low-precision on HIP. Add two helper properties that
force-cast to float32, and use them at the call sites.

Patch 04 (breakable-target-verify): in piecewise/breakable CUDA graph mode, `metadata`
is set to None before the decode/idle/target_verify/draft_extend_v2 branch, then used
directly in _get_topk_paged -> crash. Guard: if metadata is None, re-fetch via
get_attn_backend().get_indexer_metadata(); if still None, return None.

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
    assert branch_old in src, "04: decode/idle/target_verify branch not found"
    src = src.replace(branch_old, branch_new, 1)
    apply("04", True)

path.write_text(src)
print(f"[ok] {path}: " + ", ".join(changed))
