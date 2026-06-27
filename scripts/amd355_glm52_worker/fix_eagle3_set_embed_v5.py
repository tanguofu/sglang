#!/usr/bin/env python3
"""Fix v5: Add load_lm_head_from_target = True AND hot_token_id = None to
DeepseekV3ForCausalLMNextN (in deepseek_nextn.py) for EAGLE3 compatibility."""
import pathlib, sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py"
p = pathlib.Path(FILE)
if not p.exists():
    print(f"[ERROR] file not found: {FILE}"); sys.exit(1)

text = p.read_text()

old = "class DeepseekV3ForCausalLMNextN(DeepseekV3ForCausalLM):"
new = """class DeepseekV3ForCausalLMNextN(DeepseekV3ForCausalLM):
    load_lm_head_from_target = True  # EAGLE3 compat: share lm_head with target
    hot_token_id = None  # EAGLE3 compat: no hot token id (same as llama_eagle3 default)"""

if "load_lm_head_from_target = True" in text and "hot_token_id = None" in text:
    print("[FIX] already patched v5"); sys.exit(0)

# Remove previous v4 patch if present
text = text.replace(
    "class DeepseekV3ForCausalLMNextN(DeepseekV3ForCausalLM):\n    load_lm_head_from_target = True  # EAGLE3 compat: share lm_head with target",
    "class DeepseekV3ForCausalLMNextN(DeepseekV3ForCausalLM):"
)

if old not in text:
    print("[ERROR] target class not found"); sys.exit(1)

text = text.replace(old, new, 1)
p.write_text(text)
print("[FIX] applied v5: DeepseekV3ForCausalLMNextN.load_lm_head_from_target = True + hot_token_id = None")
