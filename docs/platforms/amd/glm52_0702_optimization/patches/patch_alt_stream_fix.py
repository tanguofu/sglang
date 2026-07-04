#!/usr/bin/env python3
"""Fix alt_stream on HIP - the v6 patch skip check was buggy."""
import re

GLM4 = "/sgl-workspace/sglang/python/sglang/srt/models/glm4_moe.py"

with open(GLM4, "r") as f:
    content = f.read()

old = "self.alt_stream = torch.cuda.Stream() if _is_cuda else None"
new = "self.alt_stream = torch.cuda.Stream()"

if old in content:
    content = content.replace(old, new, 1)
    with open(GLM4, "w") as f:
        f.write(content)
    print("[OK] Fixed alt_stream on HIP in glm4_moe.py")
elif new in content and old not in content:
    print("[SKIP] alt_stream already fixed")
else:
    print("[WARN] Pattern not found for alt_stream fix")
