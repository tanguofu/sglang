#!/usr/bin/env python3
"""Patch dspark_worker_v2.py: construct custom_mask for bidirectional block attention.

The training attention mask (create_dspark_attention_mask in DeepSpec common.py):
- Context tokens: all draft tokens attend to all context tokens (kv_idx < anchor_pos,
  but in inference anchor_pos ≈ seq_len so effectively all context)
- Draft tokens: bidirectional within block (q_block_id == kv_block_id)

In inference, the default is causal (each draft token only sees tokens before it).
This patch constructs a custom_mask that makes block attention bidirectional,
matching training.

Mask format (per request): draft_num * (seq_len + draft_num) bool values.
- Row i (draft token i): True for all context tokens + True for all draft tokens
  in the same block (bidirectional).
"""
import re

f = "/data/sglang_src/python/sglang/srt/speculative/dspark_worker_v2.py"
code = open(f).read()

# Find _run_draft_block and add custom_mask construction
old = """        draft_block_spec_info = DSparkVerifyInput(
            draft_token=block_ids.reshape(-1),
            positions=positions,
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            block_full_attn=int(self.block_size),
        )"""

new = """        # Construct custom_mask for bidirectional block attention (matching training).
        # Mask shape per request: draft_num * (seq_len + draft_num) bool.
        # - Context tokens (KV 0..seq_len-1): all draft tokens attend (True)
        # - Draft tokens (KV seq_len..seq_len+draft_num-1): bidirectional within block (True)
        block_size = int(self.block_size)
        custom_mask = None
        if block_size > 1:
            masks = []
            for b in range(bs):
                sl = int(prefix_lens[b].item())
                kv_len = sl + block_size
                # All True: every draft token attends to every KV token (context + block)
                # This matches training's bidirectional block attention + full context attention
                m = torch.ones(block_size, kv_len, dtype=torch.bool, device=device)
                masks.append(m.reshape(-1))
            custom_mask = torch.cat(masks, dim=0)

        draft_block_spec_info = DSparkVerifyInput(
            draft_token=block_ids.reshape(-1),
            positions=positions,
            draft_token_num=int(self.block_size),
            custom_mask=custom_mask,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            block_full_attn=int(self.block_size),
        )"""

if old in code:
    code = code.replace(old, new)
    open(f, "w").write(code)
    print("custom_mask patch applied")
else:
    print("PATTERN NOT FOUND")
