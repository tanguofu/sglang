#!/usr/bin/env python3
"""Patch dsa_backend.py: use self.qk_nope_head_dim/self.qk_rope_head_dim
instead of layer.v_head_dim/layer.head_dim-layer.v_head_dim for q splitting.
GLM-5.2 has v_head_dim=256=head_dim, so head_dim-v_head_dim=0.
Use qk_rope_head_dim (64) and qk_nope_head_dim (192) from model_config instead."""
import sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"

with open(FILE, "r") as f:
    content = f.read()

total = 0

# Fix q_nope: layer.v_head_dim -> self.qk_nope_head_dim
old1 = "q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)"
new1 = "q_nope = q.reshape(-1, layer.tp_q_head_num, self.qk_nope_head_dim)"
c = content.count(old1); content = content.replace(old1, new1)
if c: print(f"  Replaced {c}x: q_nope view->reshape + v_head_dim->qk_nope_head_dim"); total += c

# Fix q_rope: layer.head_dim - layer.v_head_dim -> self.qk_rope_head_dim
old2 = "q_rope = q_rope.view(\n                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim"
new2 = "q_rope = q_rope.reshape(\n                -1, layer.tp_q_head_num, self.qk_rope_head_dim"
c = content.count(old2); content = content.replace(old2, new2)
if c: print(f"  Replaced {c}x: q_rope view->reshape + head_dim-v_head_dim->qk_rope_head_dim"); total += c

# Fix q_rope_reshaped
old3 = "q_rope_reshaped = q_rope.view(\n                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim"
new3 = "q_rope_reshaped = q_rope.reshape(\n                -1, layer.tp_q_head_num, self.qk_rope_head_dim"
c = content.count(old3); content = content.replace(old3, new3)
if c: print(f"  Replaced {c}x: q_rope_reshaped view->reshape"); total += c

# Fix q_all: layer.head_dim -> self.qk_nope_head_dim + self.qk_rope_head_dim
old4 = "q_all = q.view(-1, layer.tp_q_head_num, layer.head_dim)"
new4 = "q_all = q.reshape(-1, layer.tp_q_head_num, self.qk_nope_head_dim + self.qk_rope_head_dim)"
c = content.count(old4); content = content.replace(old4, new4)
if c: print(f"  Replaced {c}x: q_all view->reshape + head_dim->qk_nope+qk_rope"); total += c

# Fix else branch slicing: v_head_dim -> self.qk_nope_head_dim
old5 = "q_nope = q_all[:, :, : layer.v_head_dim]\n            q_rope = q_all[:, :, layer.v_head_dim :]"
new5 = "q_nope = q_all[:, :, : self.qk_nope_head_dim]\n            q_rope = q_all[:, :, self.qk_nope_head_dim :]"
c = content.count(old5); content = content.replace(old5, new5)
if c: print(f"  Replaced {c}x: q_all slicing v_head_dim->qk_nope_head_dim"); total += c

if total == 0 and "self.qk_nope_head_dim" in content and "self.qk_rope_head_dim" in content:
    # Check if already patched
    if "q.reshape(-1, layer.tp_q_head_num, self.qk_nope_head_dim)" in content:
        print("[PATCH] dsa_backend.py already patched")
        sys.exit(0)

with open(FILE, "w") as f:
    f.write(content)
print(f"[PATCH] Successfully patched dsa_backend.py ({total} replacement(s))")
