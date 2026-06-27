#!/usr/bin/env python3
"""Fix v4: DeepseekV3ForCausalLMNextN is in deepseek_nextn.py (not deepseek_v2.py).
Add load_lm_head_from_target = True to DeepseekV3ForCausalLMNextN."""
import pathlib, sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py"
p = pathlib.Path(FILE)
if not p.exists():
    print(f"[ERROR] file not found: {FILE}"); sys.exit(1)

text = p.read_text()

old = "class DeepseekV3ForCausalLMNextN(DeepseekV3ForCausalLM):"
new = "class DeepseekV3ForCausalLMNextN(DeepseekV3ForCausalLM):\n    load_lm_head_from_target = True  # EAGLE3 compat: share lm_head with target"

if "load_lm_head_from_target = True" in text:
    print("[FIX] already patched"); sys.exit(0)

if old not in text:
    print("[ERROR] target class not found"); sys.exit(1)

text = text.replace(old, new, 1)
p.write_text(text)
print("[FIX] applied: DeepseekV3ForCausalLMNextN.load_lm_head_from_target = True (in deepseek_nextn.py)")
