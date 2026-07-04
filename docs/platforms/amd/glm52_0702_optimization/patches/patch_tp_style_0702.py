#!/usr/bin/env python3
"""Patch transformers.py to support mla_kv_a_proj TP style on 0702 image."""

TRANSFORMERS_PY = "/sgl-workspace/sglang/python/sglang/srt/models/transformers.py"

with open(TRANSFORMERS_PY, "r") as f:
    content = f.read()

OLD = '''        "moe_tp_experts": "replicate",
    }.get(style, style)'''

NEW = '''        "moe_tp_experts": "replicate",
        "mla_kv_a_proj": "replicate",
    }.get(style, style)'''

if OLD in content:
    content = content.replace(OLD, NEW, 1)
    print("[OK] Patched: mla_kv_a_proj -> replicate")
elif NEW in content:
    print("[SKIP] Already patched: mla_kv_a_proj")
else:
    print("[WARN] Pattern not found")

with open(TRANSFORMERS_PY, "w") as f:
    f.write(content)
