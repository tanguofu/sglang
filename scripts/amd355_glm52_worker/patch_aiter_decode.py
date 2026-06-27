#!/usr/bin/env python3
"""Patch dsa_backend.py for aiter decode backend: fix 3 bugs from iwiki 4022990943."""
import sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"

with open(FILE, "r") as f:
    content = f.read()

total = 0

# Bug 1: q_scale=None for FP8 -> set to ones
old = "        q_scale = None\n        kv_scale = None\n        aiter_persistent_kwargs = {}\n        if kv_cache.dtype == fp8_dtype:\n            kv_scale = torch.ones((), dtype=torch.float32, device=q_kernel.device)"
new = "        q_scale = None\n        kv_scale = None\n        aiter_persistent_kwargs = {}\n        if kv_cache.dtype == fp8_dtype:\n            q_scale = torch.ones((), dtype=torch.float32, device=q_kernel.device)\n            kv_scale = torch.ones((), dtype=torch.float32, device=q_kernel.device)"
c = content.count(old)
content = content.replace(old, new)
if c: print(f"  Bug1: Fixed q_scale=None -> ones ({c}x)"); total += c

# Bug 2: o = q.new_empty() inherits fp8 dtype -> force bfloat16
# In _forward_aiter decode path
old = "        if layer.head_dim != layer.v_head_dim:\n            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))\n        else:\n            o = torch.empty_like(q)"
new = "        if layer.head_dim != layer.v_head_dim:\n            o = torch.empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim), dtype=torch.bfloat16, device=q.device)\n        else:\n            o = torch.empty_like(q, dtype=torch.bfloat16)"
c = content.count(old)
content = content.replace(old, new)
if c: print(f"  Bug2: Fixed o=new_empty fp8 -> bfloat16 ({c}x)"); total += c

# Bug 2b: same for extend path
old = "        if layer.head_dim != layer.v_head_dim:\n            o = q.new_empty((num_tokens, layer.tp_q_head_num * layer.v_head_dim))\n        else:\n            o = torch.empty_like(q)"
new = "        if layer.head_dim != layer.v_head_dim:\n            o = torch.empty((num_tokens, layer.tp_q_head_num * layer.v_head_dim), dtype=torch.bfloat16, device=q.device)\n        else:\n            o = torch.empty_like(q, dtype=torch.bfloat16)"
c = content.count(old)
content = content.replace(old, new)
if c: print(f"  Bug2b: Fixed extend o=new_empty fp8 -> bfloat16 ({c}x)"); total += c

# Bug 2c: o_kernel = q.new_empty -> force bfloat16
old = "            o_kernel = q.new_empty(\n                (\n                    q.shape[0],\n                    layer.tp_q_head_num * self.head_repeat_factor,\n                    layer.v_head_dim,\n                )\n            )"
new = "            o_kernel = torch.empty(\n                (\n                    q.shape[0],\n                    layer.tp_q_head_num * self.head_repeat_factor,\n                    layer.v_head_dim,\n                ),\n                dtype=torch.bfloat16,\n                device=q.device,\n            )"
c = content.count(old)
content = content.replace(old, new)
if c: print(f"  Bug2c: Fixed o_kernel=new_empty fp8 -> bfloat16 ({c}x)"); total += c

# Bug 2d: same for extend path
old = "            o_kernel = q.new_empty(\n                (\n                    num_tokens,\n                    layer.tp_q_head_num * self.head_repeat_factor,\n                    layer.v_head_dim,\n                )\n            )"
new = "            o_kernel = torch.empty(\n                (\n                    num_tokens,\n                    layer.tp_q_head_num * self.head_repeat_factor,\n                    layer.v_head_dim,\n                ),\n                dtype=torch.bfloat16,\n                device=q.device,\n            )"
c = content.count(old)
content = content.replace(old, new)
if c: print(f"  Bug2d: Fixed extend o_kernel=new_empty fp8 -> bfloat16 ({c}x)"); total += c

with open(FILE, "w") as f:
    f.write(content)

print(f"\nTotal fixes applied: {total}")
if total == 0:
    print("WARNING: No fixes applied - patterns may have changed")
