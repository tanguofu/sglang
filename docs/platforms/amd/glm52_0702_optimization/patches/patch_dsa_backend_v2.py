#!/usr/bin/env python3
"""Patch dsa_backend.py: ONLY change .view() -> .reshape() for non-contiguous tensors.
Keep original dimensions: layer.v_head_dim, layer.head_dim - layer.v_head_dim, layer.head_dim.
The previous patch incorrectly changed v_head_dim -> qk_nope_head_dim which broke MLA absorb path.
"""
import sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"

with open(FILE, "r") as f:
    content = f.read()

total = 0

# 1. q_nope: .view -> .reshape (keep layer.v_head_dim)
old = "q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)"
new = "q_nope = q.reshape(-1, layer.tp_q_head_num, layer.v_head_dim)"
c = content.count(old)
content = content.replace(old, new)
if c: print(f"  Replaced {c}x: q_nope view->reshape (v_head_dim kept)"); total += c

# 2. q_rope: .view -> .reshape (keep layer.head_dim - layer.v_head_dim)
old = "q_rope = q_rope.view(\n                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim"
new = "q_rope = q_rope.reshape(\n                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim"
c = content.count(old)
content = content.replace(old, new)
if c: print(f"  Replaced {c}x: q_rope view->reshape (head_dim-v_head_dim kept)"); total += c

# 3. q_rope_reshaped: .view -> .reshape
old = "q_rope_reshaped = q_rope.view(\n                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim"
new = "q_rope_reshaped = q_rope.reshape(\n                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim"
c = content.count(old)
content = content.replace(old, new)
if c: print(f"  Replaced {c}x: q_rope_reshaped view->reshape"); total += c

# 4. q_all: .view -> .reshape (keep layer.head_dim)
old = "q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)"
new = "q_all = q.contiguous().reshape(-1, layer.tp_q_head_num, layer.head_dim)"
c = content.count(old)
content = content.replace(old, new)
if c: print(f"  Replaced {c}x: q_all contiguous().view->reshape (head_dim kept)"); total += c

# 5. q_all (non-contiguous variant): .view -> .reshape
old = "q_all = q.view(-1, layer.tp_q_head_num, layer.head_dim)"
new = "q_all = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)"
c = content.count(old)
content = content.replace(old, new)
if c: print(f"  Replaced {c}x: q_all view->reshape (head_dim kept)"); total += c

# 6. q_all slicing: keep layer.v_head_dim (revert any qk_nope_head_dim changes)
old = "q_nope = q_all[:, :, : self.qk_nope_head_dim]\n            q_rope = q_all[:, :, self.qk_nope_head_dim :]"
new = "q_nope = q_all[:, :, : layer.v_head_dim]\n            q_rope = q_all[:, :, layer.v_head_dim :]"
c = content.count(old)
content = content.replace(old, new)
if c: print(f"  Replaced {c}x: q_all slicing qk_nope_head_dim->v_head_dim (reverted)"); total += c

# 7. Revert any q.reshape with qk_nope_head_dim back to v_head_dim
old = "q_nope = q.reshape(-1, layer.tp_q_head_num, self.qk_nope_head_dim)"
new = "q_nope = q.reshape(-1, layer.tp_q_head_num, layer.v_head_dim)"
c = content.count(old)
content = content.replace(old, new)
if c: print(f"  Replaced {c}x: q_nope reshape qk_nope_head_dim->v_head_dim (reverted)"); total += c

# 8. Revert any q_rope.reshape with qk_rope_head_dim back to head_dim - v_head_dim
old = "q_rope = q_rope.reshape(\n                -1, layer.tp_q_head_num, self.qk_rope_head_dim"
new = "q_rope = q_rope.reshape(\n                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim"
c = content.count(old)
content = content.replace(old, new)
if c: print(f"  Replaced {c}x: q_rope reshape qk_rope_head_dim->head_dim-v_head_dim (reverted)"); total += c

# 9. Revert q_rope_reshaped.reshape with qk_rope_head_dim
old = "q_rope_reshaped = q_rope.reshape(\n                -1, layer.tp_q_head_num, self.qk_rope_head_dim"
new = "q_rope_reshaped = q_rope.reshape(\n                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim"
c = content.count(old)
content = content.replace(old, new)
if c: print(f"  Replaced {c}x: q_rope_reshaped reshape qk_rope_head_dim->head_dim-v_head_dim (reverted)"); total += c

# 10. Revert q.reshape with qk_nope_head_dim + qk_rope_head_dim back to head_dim
old = "q_all = q.reshape(-1, layer.tp_q_head_num, self.qk_nope_head_dim + self.qk_rope_head_dim)"
new = "q_all = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)"
c = content.count(old)
content = content.replace(old, new)
if c: print(f"  Replaced {c}x: q_all reshape qk_nope+qk_rope->head_dim (reverted)"); total += c

# Check if already correctly patched
if total == 0:
    if "q.reshape(-1, layer.tp_q_head_num, layer.v_head_dim)" in content and "self.qk_nope_head_dim" not in content:
        print("[PATCH] dsa_backend.py already correctly patched")
        sys.exit(0)

with open(FILE, "w") as f:
    f.write(content)
print(f"[PATCH] Successfully patched dsa_backend.py ({total} replacement(s))")

# Verify no qk_nope_head_dim/qk_rope_head_dim remain in q reshape lines
remaining = content.count("self.qk_nope_head_dim") + content.count("self.qk_rope_head_dim")
# These are OK in __init__ for storing the values, just not in q reshape lines
print(f"[VERIFY] Remaining qk_nope/qk_rope_head_dim references: {remaining} (expected: 2 in __init__)")
