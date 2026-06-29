#!/usr/bin/env python3
"""Add Markov head (markov_w1, markov_w2) to DFlashDraftModel for DSpark support.

The DeepSpec checkpoint saves weights as:
  - markov_head.markov_w1.weight  (nn.Embedding: [vocab_size, markov_rank])
  - markov_head.markov_w2.weight  (nn.Linear:   [vocab_size, markov_rank])

SGLang's DFlashDraftModel needs these as direct attributes so DSparkWorkerV2
can find them via getattr(draft_model, "markov_w1", None).
"""
import os, re

FILE = "/sgl-workspace/sglang/python/sglang/srt/models/dflash.py"

if not os.path.exists(FILE):
    print(f"[ERROR] {FILE} not found")
    exit(1)

with open(FILE) as f:
    content = f.read()

if "self.markov_w1" in content:
    print("[PATCH] Already patched - markov_w1 exists")
    exit(0)

# 1. Add markov_w1 and markov_w2 to __init__, after self.block_size
old_init_end = "        self.block_size = draft_config.resolve_block_size(default=16)"
new_init_end = """        self.block_size = draft_config.resolve_block_size(default=16)

        # DSpark Markov head: low-rank bias from previous token.
        # markov_w1: Embedding [vocab_size, markov_rank]
        # markov_w2: Linear     [vocab_size, markov_rank]  (weight stored transposed)
        markov_rank = int(getattr(config, "markov_rank", 0))
        if markov_rank > 0:
            vocab_size = int(getattr(config, "vocab_size", 0))
            self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
            self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)
        else:
            self.markov_w1 = None
            self.markov_w2 = None"""

if old_init_end not in content:
    print("[ERROR] Could not find init end marker")
    exit(1)

content = content.replace(old_init_end, new_init_end)

# 2. Update load_weights to handle markov_head.* weight names
old_resolve = """        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped_name = name[len("model.") :]
                if stripped_name in params_dict:
                    return stripped_name
            else:
                prefixed_name = f"model.{name}"
                if prefixed_name in params_dict:
                    return prefixed_name
            return None"""

new_resolve = """        def resolve_param_name(name: str) -> Optional[str]:
            # Map markov_head.markov_w1.weight -> markov_w1.weight
            if name.startswith("markov_head."):
                name = name[len("markov_head."):]
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped_name = name[len("model.") :]
                if stripped_name in params_dict:
                    return stripped_name
            else:
                prefixed_name = f"model.{name}"
                if prefixed_name in params_dict:
                    return prefixed_name
            return None"""

if old_resolve not in content:
    print("[ERROR] Could not find resolve_param_name")
    exit(1)

content = content.replace(old_resolve, new_resolve)

with open(FILE, "w") as f:
    f.write(content)

print("[PATCH] Added markov_w1/markov_w2 to DFlashDraftModel and updated load_weights")

# Verify syntax
import py_compile
try:
    py_compile.compile(FILE, doraise=True)
    print("[VERIFY] Syntax OK")
except py_compile.PyCompileError as e:
    print(f"[ERROR] Syntax error: {e}")
    exit(1)
