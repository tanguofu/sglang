#!/usr/bin/env python3
"""Add **kwargs to aiter_backend.py forward_extend/forward_decode for DSA topk_indices compat.

The DSA backend passes topk_indices as a kwarg, but the aiter backend's
forward_extend/forward_decode signatures don't accept **kwargs, causing
TypeError. Add **kwargs to both methods.

Idempotent.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/port-post1/base/srt_layers_attention_aiter_backend.py")
src = path.read_text()
changed = []

if "_dsa_kwargs_compat" in src:
    changed.append("_dsa_kwargs_compat: skipped")
else:
    # Patch forward_extend: add comment + **kwargs
    old_extend = "    def forward_extend(\n"
    # Find the forward_extend in the DSA-compatible class (not the base)
    # We patch ALL forward_extend occurrences that don't already have the marker
    n = src.count(old_extend)
    patched = 0
    for _ in range(n):
        idx = src.find(old_extend)
        if idx == -1:
            break
        # Check if this is already patched
        check = src[idx:idx+60]
        if "_dsa_kwargs_compat" in check:
            old_extend = "    def forward_extend(\n"
            continue
        # Find the closing paren of the signature
        # Add comment on the def line and **kwargs before the closing
        new_def = "    def forward_extend(  # _dsa_kwargs_compat\n"
        src = src[:idx] + new_def + src[idx+len(old_extend):]
        # Find the next "        )\n" after this point and add **kwargs before it
        close_idx = src.find("        )\n", idx)
        if close_idx != -1:
            src = src[:close_idx] + "        **kwargs,\n" + src[close_idx:]
            patched += 1
        old_extend = "    def forward_extend(\n"

    # Patch forward_decode similarly
    old_decode = "    def forward_decode(\n"
    n = src.count(old_decode)
    for _ in range(n):
        idx = src.find(old_decode)
        if idx == -1:
            break
        check = src[idx:idx+60]
        if "_dsa_kwargs_compat" in check:
            continue
        new_def = "    def forward_decode(  # _dsa_kwargs_compat\n"
        src = src[:idx] + new_def + src[idx+len(old_decode):]
        close_idx = src.find("        )\n", idx)
        if close_idx != -1:
            src = src[:close_idx] + "        **kwargs,\n" + src[close_idx:]
            patched += 1
        old_decode = "    def forward_decode(\n"

    changed.append(f"_dsa_kwargs_compat: {patched} methods patched")

path.write_text(src)
print(f"[ok] {path}: " + ", ".join(changed))
