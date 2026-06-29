#!/usr/bin/env python3
"""Add bonus_tokens property alias to DFlashDraftInputV2 for v0.5.14 compat.

The local dflash_info_v2.py uses `verified_id` instead of `bonus_tokens`,
but v0.5.14's overlap_utils.py reads `draft_input.bonus_tokens`.
This patch adds a property alias so both work.
"""
import os, re

FILE = "/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_info_v2.py"

if not os.path.exists(FILE):
    print(f"[ERROR] {FILE} not found")
    exit(1)

with open(FILE) as f:
    content = f.read()

if "def bonus_tokens" in content:
    print("[PATCH] Already patched - bonus_tokens alias exists")
    exit(0)

# Find the class definition and add property after the dataclass fields
# Insert before the first method definition
marker = "    def "
idx = content.find(marker)
if idx == -1:
    print("[ERROR] Could not find first method in DFlashDraftInputV2")
    exit(1)

# Find the class body start to insert the property
# We need to insert after the dataclass fields but before methods
# Find the line with the first method
lines = content.split("\n")
insert_idx = None
for i, line in enumerate(lines):
    if line.startswith("    def "):
        insert_idx = i
        break

if insert_idx is None:
    print("[ERROR] Could not find insertion point")
    exit(1)

alias_code = '''    @property
    def bonus_tokens(self):
        """Alias for verified_id (v0.5.14 overlap_utils compat)."""
        return self.verified_id

    @bonus_tokens.setter
    def bonus_tokens(self, value):
        self.verified_id = value

'''

lines.insert(insert_idx, alias_code.rstrip())
content = "\n".join(lines)

with open(FILE, "w") as f:
    f.write(content)

print("[PATCH] Added bonus_tokens property alias to DFlashDraftInputV2")

# Verify
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
import torch
d = DFlashDraftInputV2(
    topk_p=torch.empty(0),
    topk_index=torch.empty(0),
    verified_id=torch.tensor([1, 2, 3]),
    new_seq_lens=torch.empty(0),
    hidden_states=torch.empty(0),
)
print(f"[VERIFY] verified_id = {d.verified_id}")
print(f"[VERIFY] bonus_tokens = {d.bonus_tokens}")
assert torch.equal(d.verified_id, d.bonus_tokens)
print("[VERIFY] bonus_tokens alias works correctly")
