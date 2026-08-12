#!/usr/bin/env python3
"""Patch eagle_worker_v2.py: deterministic argmax for EAGLE topk=1 on ROCm.
Based on iwiki 4028171207 fix — removes `and not _is_hip` gate, adds float32 cast.
"""
import sys

filepath = "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"
with open(filepath, 'r') as f:
    content = f.read()

if "deterministic torch.argmax on both CUDA and ROCm" in content:
    print("Already patched, skipping")
    sys.exit(0)

original = content

# Site 1: draft_forward topk=1
site1_old = """            elif self.topk == 1 and not _is_hip:
                if _is_cuda:
                    # The positions advance is fused into the kernel.
                    topk_p, topk_index = draft_topk1_postprocess(
                        logits_output.next_token_logits,
                        forward_batch.positions,
                        draft_tokens_topk1,
                        i + 1,
                    )
                else:
                    topk_index = torch.argmax(
                        logits_output.next_token_logits, dim=-1, keepdim=True
                    )
                    topk_p = torch.ones_like(topk_index, dtype=torch.float32)
                    forward_batch.positions.add_(1)"""

site1_new = """            elif self.topk == 1:
                # FIX: Use deterministic torch.argmax on both CUDA and ROCm.
                # Cast to float32 for stable tie-breaking on FP8 logits.
                # At long contexts, non-deterministic fast_topk causes repetition.
                _logits_f32 = logits_output.next_token_logits.to(torch.float32)
                topk_index = torch.argmax(_logits_f32, dim=-1, keepdim=True)
                topk_p = torch.ones_like(topk_index, dtype=torch.float32)
                forward_batch.positions.add_(1)"""

if site1_old in content:
    content = content.replace(site1_old, site1_new)
    print("Site 1 (draft_forward): patched")
else:
    print("Site 1: exact pattern not found")
    # Fallback: just replace the gate condition
    old_gate = "elif self.topk == 1 and not _is_hip:"
    if old_gate in content:
        content = content.replace(old_gate, "elif self.topk == 1:", 1)
        print("Site 1: gate replaced (fallback)")

# Site 2: target_verify topk=1
site2_old = """        elif self.topk == 1 and not _is_hip:
            # Gated to CUDA: see #26358 — ROCm's argmax tie-break corrupts
            # MTP draft selection on FP8 logits.
            ret_topk_index = torch.argmax(
                draft_logits_output.next_token_logits, dim=-1, keepdim=True
            )
            ret_topk_p = torch.ones_like(ret_topk_index, dtype=torch.float32)
            ret_draft_probs = None"""

site2_new = """        elif self.topk == 1:
            # FIX: Use deterministic torch.argmax on both CUDA and ROCm.
            # Cast to float32 for stable tie-breaking on FP8 logits.
            _logits_f32 = draft_logits_output.next_token_logits.to(torch.float32)
            ret_topk_index = torch.argmax(_logits_f32, dim=-1, keepdim=True)
            ret_topk_p = torch.ones_like(ret_topk_index, dtype=torch.float32)
            ret_draft_probs = None"""

if site2_old in content:
    content = content.replace(site2_old, site2_new)
    print("Site 2 (target_verify): patched")
else:
    print("Site 2: exact pattern not found")
    old_gate = "elif self.topk == 1 and not _is_hip:"
    if old_gate in content:
        content = content.replace(old_gate, "elif self.topk == 1:", 1)
        print("Site 2: gate replaced (fallback)")

if content != original:
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"\nFile written successfully")
    remaining = content.count("and not _is_hip")
    has_float32 = content.count("_logits_f32")
    print(f"Remaining 'and not _is_hip': {remaining}")
    print(f"Float32 casts added: {has_float32}")
else:
    print("No changes made")
