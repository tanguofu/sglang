#!/usr/bin/env python3
"""Fix dual-stream kw UnboundLocalError: kw must be assigned outside the if block."""
INDEXER = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"

with open(INDEXER, "r") as f:
    content = f.read()

# Bug: in dual-stream path, kw is inside the if block
old = """        if isinstance(x, tuple) and len(x) == 3:
            x = x[2]
            kw, _ = self.wk_weights_proj(x)
        key, weights_raw = kw.split"""

new = """        if isinstance(x, tuple) and len(x) == 3:
            x = x[2]
        kw, _ = self.wk_weights_proj(x)
        key, weights_raw = kw.split"""

if old in content:
    content = content.replace(old, new, 1)
    with open(INDEXER, "w") as f:
        f.write(content)
    print("[OK] Fixed dual-stream kw UnboundLocalError")
elif new in content:
    print("[SKIP] Already fixed")
else:
    print("[WARN] Pattern not found")
