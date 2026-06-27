#!/usr/bin/env python3
"""Apply Bug B one-line fix inside container: add _USE_FUSED_METADATA_COPY guard
to the multi-backend fused-copy path at dsa_backend.py ~line 2604."""
import re, pathlib
p = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py")
text = p.read_text()
old = "if self.speculative_num_steps > 3:"
new = "if self.speculative_num_steps > 3 and _USE_FUSED_METADATA_COPY:"
if old not in text:
    # already patched or pattern changed
    if new in text:
        print("[FIX] already patched")
    else:
        print("[FIX] ERROR: target string not found")
        import sys; sys.exit(1)
else:
    # only patch the multi-backend one (line ~2604), not others
    # the single-backend path uses `if _USE_FUSED_METADATA_COPY:` (different string)
    text = text.replace(old, new, 1)
    p.write_text(text)
    print("[FIX] applied: guarded multi-backend path with _USE_FUSED_METADATA_COPY")
