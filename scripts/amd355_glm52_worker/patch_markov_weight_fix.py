#!/usr/bin/env python3
"""Fix: pass markov_w1.weight and markov_w2.weight tensors, not nn.Module objects."""
import os

FILE = "/sgl-workspace/sglang/python/sglang/srt/speculative/dspark_worker_v2.py"

if not os.path.exists(FILE):
    print(f"[ERROR] {FILE} not found")
    exit(1)

with open(FILE) as f:
    content = f.read()

if "markov_w1.weight" in content:
    print("[PATCH] Already patched")
    exit(0)

old = """        markov_w1 = getattr(draft_model, "markov_w1", None)
        markov_w2 = getattr(draft_model, "markov_w2", None)

        if markov_w1 is not None and markov_w2 is not None:
            self._dspark_markov_w1 = markov_w1
            self._dspark_markov_w2 = markov_w2
            if self.tp_rank == 0:
                logger.info(
                    "DSPARK: Markov head resolved from draft model. "
                    "w1 shape=%s, w2 shape=%s",
                    tuple(markov_w1.shape),
                    tuple(markov_w2.shape),
                )
            return True"""

new = """        markov_w1_mod = getattr(draft_model, "markov_w1", None)
        markov_w2_mod = getattr(draft_model, "markov_w2", None)

        if markov_w1_mod is not None and markov_w2_mod is not None:
            self._dspark_markov_w1 = markov_w1_mod.weight if hasattr(markov_w1_mod, "weight") else markov_w1_mod
            self._dspark_markov_w2 = markov_w2_mod.weight if hasattr(markov_w2_mod, "weight") else markov_w2_mod
            if self.tp_rank == 0:
                logger.info(
                    "DSPARK: Markov head resolved from draft model. "
                    "w1 shape=%s, w2 shape=%s",
                    tuple(self._dspark_markov_w1.shape),
                    tuple(self._dspark_markov_w2.shape),
                )
            return True"""

if old not in content:
    print("[ERROR] Could not find target code")
    exit(1)

content = content.replace(old, new)

with open(FILE, "w") as f:
    f.write(content)

print("[PATCH] Fixed markov_w1/w2 to use .weight tensors")

import py_compile
try:
    py_compile.compile(FILE, doraise=True)
    print("[VERIFY] Syntax OK")
except py_compile.PyCompileError as e:
    print(f"[ERROR] Syntax error: {e}")
    exit(1)
