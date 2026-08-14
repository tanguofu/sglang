#!/usr/bin/env python3
"""Port patch 06b (frozen_kv_mtp_worker_v2 seq_lens_cpu fast path) to post1.

Semantic change: avoid a GPU sync for `batch.seq_lens_sum` when a CPU copy of
seq_lens is already available. Base unconditionally does `torch.sum(batch.seq_lens).item()`
(always a GPU->CPU sync); use `batch.seq_lens_cpu.sum().item()` when present, matching the
existing pattern at lines 301-302.

(The rest of the branch/base diff is the branch being OLDER than post1 — parallel-state
API, server_args.override, _get_plan_stream — those are NOT ported.)

Idempotent.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/port-post1/base/srt_speculative_frozen_kv_mtp_worker_v2.py")
src = path.read_text()

old = "        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()\n"
new = (
    "        if batch.seq_lens_cpu is not None:\n"
    "            batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()\n"
    "        else:\n"
    "            batch.seq_lens_sum = torch.sum(batch.seq_lens).item()\n"
)

if "batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()" in src:
    print(f"[skip] {path}: patch 06b already applied")
    sys.exit(0)

assert src.count(old) == 1, f"06b: expected 1 occurrence, found {src.count(old)}"
src = src.replace(old, new, 1)
path.write_text(src)
print(f"[ok] {path}: patch 06b applied")
