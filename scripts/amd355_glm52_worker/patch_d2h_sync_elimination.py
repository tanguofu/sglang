#!/usr/bin/env python3
"""Patch to eliminate D2H sync in DSA backend and FrozenKV MTP worker.

Root cause: dsa_backend.py:691 calls forward_batch.seq_lens.max().item()
when seq_lens_cpu is None, causing a 13.5ms D2H sync per draft_extend step.

Fix: Use pre-computed seq_lens_sum (int) as upper bound for max_seqlen_k
when seq_lens_cpu is unavailable. For bs=1, sum == max (exact); for bs>1,
sum > max (safe over-estimate, slightly larger page table slice).
"""
import os, re

SGLANG_DIR = "/sgl-workspace/sglang/python/sglang/srt"

# --- Patch 1: dsa_backend.py - eliminate .item() on GPU tensor ---
dsa_path = os.path.join(SGLANG_DIR, "layers/attention/dsa_backend.py")
with open(dsa_path, "r") as f:
    content = f.read()

old_block = """        if forward_batch.seq_lens_cpu is not None:
            max_seqlen_k = int(
                forward_batch.seq_lens_cpu.max().item() + draft_token_num
            )
        else:
            # needs_cpu_seq_lens=False nulls the host mirror for spec-v2 relay
            # batches; graph replay uses the static page-table width, so only this
            # eager (e.g. over-capture-bs) fallback needs a length here.
            max_seqlen_k = int(forward_batch.seq_lens.max().item()) + draft_token_num"""

new_block = """        if forward_batch.seq_lens_cpu is not None:
            max_seqlen_k = int(
                forward_batch.seq_lens_cpu.max().item() + draft_token_num
            )
        elif forward_batch.seq_lens_sum is not None:
            # Avoid D2H sync: use pre-computed sum as upper bound for max.
            # For bs=1, sum == max (exact); for bs>1, sum > max (safe over-estimate).
            max_seqlen_k = forward_batch.seq_lens_sum + draft_token_num
        else:
            # needs_cpu_seq_lens=False nulls the host mirror for spec-v2 relay
            # batches; graph replay uses the static page-table width, so only this
            # eager (e.g. over-capture-bs) fallback needs a length here.
            max_seqlen_k = int(forward_batch.seq_lens.max().item()) + draft_token_num"""

if old_block in content:
    content = content.replace(old_block, new_block)
    with open(dsa_path, "w") as f:
        f.write(content)
    print(f"[OK] Patched dsa_backend.py - eliminated GPU .item() D2H sync")
else:
    print(f"[SKIP] dsa_backend.py - pattern not found (may already be patched)")

# --- Patch 2: frozen_kv_mtp_worker_v2.py - use seq_lens_cpu for sum ---
mtp_path = os.path.join(SGLANG_DIR, "speculative/frozen_kv_mtp_worker_v2.py")
with open(mtp_path, "r") as f:
    content = f.read()

old_line = "        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()"
new_line = """        if batch.seq_lens_cpu is not None:
            batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()
        else:
            batch.seq_lens_sum = torch.sum(batch.seq_lens).item()"""

if old_line in content:
    content = content.replace(old_line, new_line, 1)
    with open(mtp_path, "w") as f:
        f.write(content)
    print(f"[OK] Patched frozen_kv_mtp_worker_v2.py - use seq_lens_cpu for sum")
else:
    print(f"[SKIP] frozen_kv_mtp_worker_v2.py - pattern not found (may already be patched)")

print("\nDone. Restart SGLang to apply.")
