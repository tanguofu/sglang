#!/usr/bin/env python3
"""
Patch GlmMoeDsaConfig in transformers to fix qk_rope_head_dim override bug.

Root cause: attribute_map has "head_dim": "qk_rope_head_dim", which causes
the head_dim value (192) from config.json to overwrite qk_rope_head_dim (64).

This makes fused_qkv_a_proj_with_mqa expect size [2752, 6144] instead of
the correct [2624, 6144], causing weight loading failure.

Fix: Remove the "head_dim" -> "qk_rope_head_dim" mapping from attribute_map.
SGLang's model_config.py already sets head_dim=256 for MLA architectures,
so this mapping is not needed and only causes harm for GLM-5.2 DSA models.
"""

import sys
import os

CONFIG_FILE = "/opt/venv/lib/python3.10/site-packages/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py"

def patch():
    if not os.path.exists(CONFIG_FILE):
        print(f"[ERROR] Config file not found: {CONFIG_FILE}")
        sys.exit(1)

    with open(CONFIG_FILE, "r") as f:
        content = f.read()

    # Check if already patched
    if "# REMOVED: causes qk_rope_head_dim override" in content:
        print("[PATCH] Already patched - attribute_map head_dim mapping already removed")
        return

    # The attribute_map line to remove
    old_line = '"head_dim": "qk_rope_head_dim",'
    new_line = '# "head_dim": "qk_rope_head_dim",  # REMOVED: causes qk_rope_head_dim override by head_dim value'

    if old_line not in content:
        print(f"[ERROR] Could not find target line in config file")
        print("[ERROR] Expected to find: " + old_line)
        sys.exit(1)

    content = content.replace(old_line, new_line)

    with open(CONFIG_FILE, "w") as f:
        f.write(content)

    print("[PATCH] Successfully removed head_dim -> qk_rope_head_dim from GlmMoeDsaConfig.attribute_map")

    # Verify the fix
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("/data/models/GLM-5.2-FP8", trust_remote_code=True)
    print(f"[VERIFY] qk_rope_head_dim = {getattr(config, 'qk_rope_head_dim', 'NOT SET')}")
    print(f"[VERIFY] qk_nope_head_dim = {getattr(config, 'qk_nope_head_dim', 'NOT SET')}")
    print(f"[VERIFY] head_dim = {getattr(config, 'head_dim', 'NOT SET')}")
    print(f"[VERIFY] qk_head_dim = {getattr(config, 'qk_head_dim', 'NOT SET')}")

    expected_rope = 64
    actual_rope = getattr(config, 'qk_rope_head_dim', -1)
    if actual_rope == expected_rope:
        print("[VERIFY] SUCCESS: qk_rope_head_dim is now 64 (correct!)")
    else:
        print(f"[VERIFY] FAILED: qk_rope_head_dim is {actual_rope}, expected {expected_rope}")
        sys.exit(1)

if __name__ == "__main__":
    patch()
