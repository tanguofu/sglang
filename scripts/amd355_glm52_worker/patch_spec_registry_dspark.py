#!/usr/bin/env python3
"""Add is_dspark() to CustomSpecAlgo in spec_registry.py."""
import os

FILE = "/sgl-workspace/sglang/python/sglang/srt/speculative/spec_registry.py"
with open(FILE) as f:
    content = f.read()

if "is_dspark" in content:
    print("[PATCH] Already has is_dspark")
    exit(0)

content = content.replace(
    "    def is_dflash(self) -> bool:\n        return False",
    "    def is_dflash(self) -> bool:\n        return False\n\n    def is_dspark(self) -> bool:\n        return False"
)

with open(FILE, "w") as f:
    f.write(content)

print("[PATCH] Added is_dspark to spec_registry.py")

import py_compile
py_compile.compile(FILE, doraise=True)
print("[VERIFY] Syntax OK")
