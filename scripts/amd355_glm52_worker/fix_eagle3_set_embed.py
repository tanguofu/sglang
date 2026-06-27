#!/usr/bin/env python3
"""Fix: EAGLE3 worker calls set_embed on draft model, but NextN draft (DeepseekV3ForCausalLMNextN)
doesn't have set_embed (only has set_embed_and_head).
Fix: Add load_lm_head_from_target = True to DeepseekV3ForCausalLMNextN, so EAGLE3 worker
uses set_embed_and_head (shared lm_head) instead of set_embed (separate lm_head).
"""
import pathlib, sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py"
p = pathlib.Path(FILE)
if not p.exists():
    print(f"[ERROR] file not found: {FILE}"); sys.exit(1)

text = p.read_text()

# Find DeepseekV3ForCausalLMNextN class and add load_lm_head_from_target = True
old = "class DeepseekV3ForCausalLMNextN(DeepseekV3ForCausalLM):"
new = "class DeepseekV3ForCausalLMNextN(DeepseekV3ForCausalLM):\n    load_lm_head_from_target = True  # EAGLE3 compat: share lm_head with target"

if "load_lm_head_from_target = True" in text and "DeepseekV3ForCausalLMNextN" in text:
    # Check if already in the right place
    if "class DeepseekV3ForCausalLMNextN(DeepseekV3ForCausalLM):\n    load_lm_head_from_target" in text:
        print("[FIX] already patched"); sys.exit(0)

if old not in text:
    print("[ERROR] target class not found"); sys.exit(1)

text = text.replace(old, new, 1)
p.write_text(text)
print("[FIX] applied: DeepseekV3ForCausalLMNextN.load_lm_head_from_target = True (EAGLE3 uses set_embed_and_head)")
