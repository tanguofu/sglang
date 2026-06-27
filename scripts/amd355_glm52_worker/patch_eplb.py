#!/usr/bin/env python3
"""Patch EPLB expert_distribution.py to use moe_ep_group instead of world group for reduce.

Bug #3: torch.distributed.reduce uses world group (includes PP ranks),
causing deadlock when PP>1 or DP attention with EP.
"""
import os, sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/eplb/expert_distribution.py"

def patch():
    if not os.path.exists(TARGET):
        print(f"[ERROR] {TARGET} not found")
        sys.exit(1)
    
    with open(TARGET, "r") as f:
        content = f.read()
    
    if "get_moe_ep_group" in content and "group=get_moe_ep_group" in content:
        print("[PATCH] EPLB already patched")
        return
    
    old = "torch.distributed.reduce(\n            gpu_physical_count, dst=0, op=torch.distributed.ReduceOp.SUM\n        )"
    new = "from sglang.srt.distributed.parallel_state import get_moe_ep_group\n        torch.distributed.reduce(\n            gpu_physical_count, dst=0, op=torch.distributed.ReduceOp.SUM,\n            group=get_moe_ep_group().group\n        )"
    
    if old not in content:
        print("[ERROR] Could not find target pattern in expert_distribution.py")
        sys.exit(1)
    
    content = content.replace(old, new)
    with open(TARGET, "w") as f:
        f.write(content)
    print("[PATCH] EPLB reduce group fix applied successfully")

if __name__ == "__main__":
    patch()
