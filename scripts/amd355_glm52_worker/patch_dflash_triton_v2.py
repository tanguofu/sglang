#!/usr/bin/env python3
"""Patch _prepare_dflash_draft_block_unchecked to accept verified_id.

Fix: rename bonus_tokens -> verified_id in the function signature and body.
"""
import os

FILE = "/sgl-workspace/sglang/python/sglang/srt/speculative/triton_ops/dflash.py"

if not os.path.exists(FILE):
    print(f"[ERROR] {FILE} not found")
    exit(1)

with open(FILE) as f:
    content = f.read()

# Check if already properly patched (no syntax error version)
if "verified_id: torch.Tensor,\n    prefix_lens" in content:
    print("[PATCH] Already patched correctly")
    exit(0)

# Revert any bad patch first
content = content.replace(
    "def _prepare_dflash_draft_block_unchecked(\n    bonus_tokens: torch.Tensor = None,\n    verified_id: torch.Tensor = None,",
    "def _prepare_dflash_draft_block_unchecked(\n    bonus_tokens: torch.Tensor,"
)
content = content.replace(
    "    if bonus_tokens is None and verified_id is not None:\n        bonus_tokens = verified_id\n    batch_size = int(bonus_tokens.numel())",
    "    batch_size = int(bonus_tokens.numel())"
)

# Now apply the correct patch: rename bonus_tokens to verified_id in the function
old_sig = "def _prepare_dflash_draft_block_unchecked(\n    bonus_tokens: torch.Tensor,"
new_sig = "def _prepare_dflash_draft_block_unchecked(\n    verified_id: torch.Tensor,"

if old_sig not in content:
    print("[ERROR] Could not find function signature to patch")
    # Maybe already partially patched, try to find it
    if "bonus_tokens: torch.Tensor = None" in content:
        content = content.replace(
            "def _prepare_dflash_draft_block_unchecked(\n    bonus_tokens: torch.Tensor = None,\n    verified_id: torch.Tensor = None,",
            "def _prepare_dflash_draft_block_unchecked(\n    verified_id: torch.Tensor,"
        )
        content = content.replace(
            "    if bonus_tokens is None and verified_id is not None:\n        bonus_tokens = verified_id\n    batch_size = int(bonus_tokens.numel())",
            "    batch_size = int(verified_id.numel())"
        )
    else:
        exit(1)
else:
    content = content.replace(old_sig, new_sig)

# Replace all references to bonus_tokens within the function body
# Only in the _prepare_dflash_draft_block_unchecked function
# Find the function and replace bonus_tokens -> verified_id within it
lines = content.split("\n")
in_func = False
func_indent = 0
new_lines = []
for i, line in enumerate(lines):
    if "def _prepare_dflash_draft_block_unchecked(" in line:
        in_func = True
        func_indent = len(line) - len(line.lstrip())
        new_lines.append(line)
        continue
    if in_func:
        # Check if we've left the function (next def at same or lower indent)
        stripped = line.lstrip()
        if stripped and not stripped.startswith("#") and len(line) - len(line.lstrip()) <= func_indent and (stripped.startswith("def ") or stripped.startswith("class ")):
            in_func = False
            new_lines.append(line)
            continue
        # Replace bonus_tokens with verified_id within the function
        line = line.replace("bonus_tokens", "verified_id")
        new_lines.append(line)
    else:
        new_lines.append(line)

content = "\n".join(new_lines)

with open(FILE, "w") as f:
    f.write(content)

print("[PATCH] Renamed bonus_tokens -> verified_id in _prepare_dflash_draft_block_unchecked")

# Verify syntax
import py_compile
try:
    py_compile.compile(FILE, doraise=True)
    print("[VERIFY] Syntax OK")
except py_compile.PyCompileError as e:
    print(f"[ERROR] Syntax error: {e}")
    exit(1)
