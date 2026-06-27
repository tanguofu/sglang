#!/usr/bin/env python3
"""Fix v2: EAGLE3 worker calls set_embed on draft model, but GLM-5.2's NextN draft
(Glm4MoeForCausalLMNextN) doesn't have set_embed (only has set_embed_and_head from parent).
Fix: Add load_lm_head_from_target = True to Glm4MoeForCausalLMNextN, so EAGLE3 worker
uses set_embed_and_head (shared lm_head) instead of set_embed (separate lm_head)."""
import pathlib, sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/models/glm4_moe_nextn.py"
p = pathlib.Path(FILE)
if not p.exists():
    print(f"[ERROR] file not found: {FILE}"); sys.exit(1)

text = p.read_text()

old = "class Glm4MoeForCausalLMNextN(Glm4MoeForCausalLM):"
new = "class Glm4MoeForCausalLMNextN(Glm4MoeForCausalLM):\n    load_lm_head_from_target = True  # EAGLE3 compat: share lm_head with target"

if "load_lm_head_from_target = True" in text:
    print("[FIX] already patched"); sys.exit(0)

if old not in text:
    print("[ERROR] target class not found"); sys.exit(1)

text = text.replace(old, new, 1)
p.write_text(text)
print("[FIX] applied: Glm4MoeForCausalLMNextN.load_lm_head_from_target = True")
