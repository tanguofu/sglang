#!/usr/bin/env python3
"""Fix position offset: prefix_lens -> prefix_lens - 1 to match training's anchor_pos.

Training: positions = anchor_pos + [0..block_size-1]
  where anchor_pos = position of the last committed token = seq_len - 1
Inference: positions = prefix_lens + [0..block_size-1]
  where prefix_lens = seq_len (next token position)

Fix: positions = (prefix_lens - 1) + [0..block_size-1] = prefix_lens - 1 + [0..block_size-1]
"""
f = "/data/sglang_src/python/sglang/srt/speculative/dspark_worker_v2.py"
code = open(f).read()

old = "        positions_2d = prefix_lens.unsqueeze(1) + self._block_pos_offsets"
new = "        # Fix: use prefix_lens - 1 to match training's anchor_pos semantics\n        # Training: positions = anchor_pos + offsets, where anchor_pos = seq_len - 1\n        # Inference was: prefix_lens + offsets (= seq_len + offsets, off by 1)\n        positions_2d = (prefix_lens - 1).unsqueeze(1) + self._block_pos_offsets"

if old in code:
    code = code.replace(old, new)
    open(f, "w").write(code)
    print("position offset fix applied")
else:
    print("PATTERN NOT FOUND")
