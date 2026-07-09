#!/usr/bin/env python3
"""Patch Glm5DSparkMLAAttention.forward to use separate KV projection (matching inference kv_from_hidden).

Training currently: kv_a_proj_with_mqa(cat([target, draft])) — project concatenated
Inference:           kv_from_hidden(target) + draft KV via normal forward — separate

Fix: project target and draft SEPARATELY, then concatenate:
  k_target = kv_a_proj_with_mqa(target)  → matches kv_from_hidden
  k_draft  = kv_a_proj_with_mqa(draft)   → matches normal forward
  k = cat([k_target, k_draft])

This is mathematically equivalent for linear layers, but aligns the code path
with inference (kv_from_hidden processes target separately).
"""
import re

f = "/data/DeepSpec/deepspec/modeling/dspark/glm5/modeling.py"
code = open(f).read()

old = """        # --- KV projection (from context + draft) ---
        full_kv_input = torch.cat(
            [target_hidden_states, hidden_states], dim=1
        )  # [bsz, ctx_len + draft_len, hidden_size]
        compressed_kv = self.kv_a_proj_with_mqa(
            full_kv_input
        )  # [bsz, kv_len, kv_lora_rank + qk_rope_head_dim]
        k_nope_compressed, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        k_nope_and_v = self.kv_b_proj(
            self.kv_a_layernorm(k_nope_compressed)
        )  # [bsz, kv_len, num_heads * (qk_nope_head_dim + v_head_dim)]"""

new = """        # --- KV projection (separate target + draft, matching inference kv_from_hidden) ---
        # Target KV: project separately (matches sglang kv_from_hidden path)
        compressed_kv_ctx = self.kv_a_proj_with_mqa(
            target_hidden_states
        )  # [bsz, ctx_len, kv_lora_rank + qk_rope_head_dim]
        # Draft KV: project separately (matches sglang normal forward path)
        compressed_kv_draft = self.kv_a_proj_with_mqa(
            hidden_states
        )  # [bsz, draft_len, kv_lora_rank + qk_rope_head_dim]
        # Concatenate after projection (mathematically equivalent to projecting cat,
        # but aligns code path with inference kv_from_hidden)
        compressed_kv = torch.cat(
            [compressed_kv_ctx, compressed_kv_draft], dim=1
        )  # [bsz, kv_len, kv_lora_rank + qk_rope_head_dim]
        k_nope_compressed, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        k_nope_and_v = self.kv_b_proj(
            self.kv_a_layernorm(k_nope_compressed)
        )  # [bsz, kv_len, num_heads * (qk_nope_head_dim + v_head_dim)]"""

if old in code:
    code = code.replace(old, new)
    open(f, "w").write(code)
    print("attention fix applied: separate KV projection (matching kv_from_hidden)")
else:
    print("PATTERN NOT FOUND — checking if already applied")
    if "compressed_kv_ctx" in code:
        print("already applied")
    else:
        print("ERROR: pattern not found and not already applied")
