#!/usr/bin/env python3
"""Port patch 4.1 (transformers.py mla_kv_a_proj TP style) to post1.

Semantic change: register the `mla_kv_a_proj` weight as a `replicate` TP style in the
style-normalization map. Single-line addition.

(The branch version also uses the removed get_global_server_args in two other spots —
those are NOT ported; post1's get_server_args() is correct and stays.)

Idempotent.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/port-post1/base/srt_models_transformers.py")
src = path.read_text()

anchor = '        "moe_tp_experts": "replicate",\n    }.get(style, style)'
addition = '        "moe_tp_experts": "replicate",\n        "mla_kv_a_proj": "replicate",\n    }.get(style, style)'

if "mla_kv_a_proj" in src:
    print(f"[skip] {path}: patch 4.1 already applied")
    sys.exit(0)

assert anchor in src, "4.1: anchor not found"
src = src.replace(anchor, addition, 1)
path.write_text(src)
print(f"[ok] {path}: patch 4.1 applied")
