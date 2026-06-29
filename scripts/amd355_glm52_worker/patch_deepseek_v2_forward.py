#!/usr/bin/env python3
"""Fix deepseek_v2.py forward to check isinstance before unpacking hidden_states."""
import os

FILE = "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py"

if not os.path.exists(FILE):
    print(f"[ERROR] {FILE} not found")
    exit(1)

with open(FILE) as f:
    content = f.read()

old = """        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states"""

new = """        if self.capture_aux_hidden_states:
            if isinstance(hidden_states, (tuple, list)):
                hidden_states, aux_hidden_states = hidden_states"""

if old not in content:
    if "isinstance(hidden_states, (tuple, list))" in content:
        print("[PATCH] Already patched")
        exit(0)
    print("[ERROR] Could not find target code")
    exit(1)

content = content.replace(old, new)

with open(FILE, "w") as f:
    f.write(content)

print("[PATCH] Fixed deepseek_v2.py forward unpacking")

import py_compile
try:
    py_compile.compile(FILE, doraise=True)
    print("[VERIFY] Syntax OK")
except py_compile.PyCompileError as e:
    print(f"[ERROR] Syntax error: {e}")
    exit(1)
