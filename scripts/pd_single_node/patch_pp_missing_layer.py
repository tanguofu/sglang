#!/usr/bin/env python3
"""
Patch SGLang's deepseek_v2.py to fix PPMissingLayer.embedding_dim AttributeError.

When PP > 1, the second pipeline stage has self.embed_tokens as PPMissingLayer,
which doesn't have an 'embedding_dim' attribute. The code checks:
    self.embed_tokens.embedding_dim == 7168
which raises AttributeError on PP1 ranks.

Fix: Use getattr with a default value so the check returns False gracefully.
"""

import sys
import os

MODEL_FILE = "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py"

def patch():
    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] Model file not found: {MODEL_FILE}")
        sys.exit(1)

    with open(MODEL_FILE, "r") as f:
        content = f.read()

    # Check if already patched
    if "getattr(self.embed_tokens, 'embedding_dim'" in content:
        print("[PATCH] Already patched - PPMissingLayer.embedding_dim fix already applied")
        return

    # Patch 1: Line ~2290 - the if condition check
    old1 = "and self.embed_tokens.embedding_dim == 7168"
    new1 = "and getattr(self.embed_tokens, 'embedding_dim', 0) == 7168"

    if old1 not in content:
        print(f"[ERROR] Could not find target string 1: {old1}")
        sys.exit(1)

    content = content.replace(old1, new1)
    print("[PATCH] Patched condition check: getattr(self.embed_tokens, 'embedding_dim', 0) == 7168")

    # Patch 2: Line ~2322 - the function call argument
    old2 = "                    self.embed_tokens.embedding_dim,\n                )"
    new2 = "                    getattr(self.embed_tokens, 'embedding_dim', config.hidden_size),\n                )"

    if old2 not in content:
        print(f"[WARNING] Could not find target string 2 (may have different indentation)")
        # Try a more flexible match
        old2_alt = "self.embed_tokens.embedding_dim,"
        # Only replace if it's not already the getattr version
        if old2_alt in content and "getattr(self.embed_tokens, 'embedding_dim'" not in content:
            content = content.replace(old2_alt, "getattr(self.embed_tokens, 'embedding_dim', config.hidden_size),", 1)
            print("[PATCH] Patched function call argument (flexible match)")
        else:
            print("[WARNING] Could not patch function call argument - may not be needed")
    else:
        content = content.replace(old2, new2)
        print("[PATCH] Patched function call argument: getattr(self.embed_tokens, 'embedding_dim', config.hidden_size)")

    with open(MODEL_FILE, "w") as f:
        f.write(content)

    print("[PATCH] Successfully patched PPMissingLayer.embedding_dim fix")

if __name__ == "__main__":
    patch()
