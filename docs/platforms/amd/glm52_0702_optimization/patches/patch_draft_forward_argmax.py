#!/usr/bin/env python3
"""Patch eagle_worker_v2.py: fix draft_forward per-step argmax gate on ROCm.

The patch_deterministic_argmax.py only patches the draft_extend path (line ~915).
The draft_forward path (line ~667) still has `elif self.topk == 1 and not _is_hip:`
which forces HIP to use fast_topk(softmax(logits), 1) instead of torch.argmax.

This inconsistency means:
- Initial draft token (draft_extend): torch.argmax(logits.to(float32)) — patched
- Per-step draft tokens (draft_forward): torch.max(softmax(logits)) — NOT patched

The per-step tokens are where error compounding happens (steps 1, 2, ...).
Using softmax + max instead of direct argmax introduces numerical differences
that compound across steps, degrading accept rate especially for MTP3+.

Fix: Apply the same deterministic argmax (float32 cast + epsilon) to draft_forward.
"""
import sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"

with open(FILE, "r") as f:
    content = f.read()

# draft_forward path (line ~667): topk_index / topk_p / logits_output
old_forward = """            elif self.topk == 1 and not _is_hip:
                topk_index = torch.argmax(
                    logits_output.next_token_logits, dim=-1, keepdim=True
                )
                topk_p = torch.ones_like(topk_index, dtype=torch.float32)"""

new_forward = """            elif self.topk == 1:
                # Use deterministic torch.argmax on both CUDA and ROCm (matches draft_extend).
                # Cast to float32 for stable tie-breaking on FP8 logits.
                _logits_f32 = logits_output.next_token_logits.to(torch.float32)
                topk_index = torch.argmax(_logits_f32, dim=-1, keepdim=True)
                topk_p = torch.ones_like(topk_index, dtype=torch.float32)"""

patched = False
if old_forward in content:
    content = content.replace(old_forward, new_forward, 1)
    patched = True
elif new_forward in content:
    print("[SKIP] draft_forward argmax already patched")
    sys.exit(0)
else:
    print("[WARN] draft_forward pattern not found — checking for variant")
    # Check if the _is_hip gate was already removed by another patch
    if "elif self.topk == 1:\n                # Use deterministic torch.argmax on both CUDA and ROCm (matches draft_extend)." in content:
        print("[SKIP] draft_forward argmax already patched (variant)")
        sys.exit(0)
    print("[ERROR] Cannot find draft_forward argmax pattern")
    sys.exit(1)

if patched:
    with open(FILE, "w") as f:
        f.write(content)
    print("[OK] Patched draft_forward per-step argmax gate for ROCm")
