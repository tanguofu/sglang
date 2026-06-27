#!/usr/bin/env python3
"""Fix: DSA unconditionally overrides page_size to 64, even when user explicitly sets --page-size 1.
This prevents EAGLE3 + topk>1 + page_size=1 from working (topk>1 + page_size>1 + DSA = rejected).
Fix: only set page_size=64 when user hasn't explicitly set it (page_size is None)."""
import pathlib, sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/server_args.py"
p = pathlib.Path(FILE)
if not p.exists():
    print(f"[ERROR] file not found: {FILE}"); sys.exit(1)

text = p.read_text()

# Fix: only override page_size when it's None (not explicitly set by user)
old = '''                    else:
                        self.page_size = 64
                        logger.warning("Setting page size to 64 for DeepSeek DSA.")'''
new = '''                    else:
                        if self.page_size is None:
                            self.page_size = 64
                            logger.warning("Setting page size to 64 for DeepSeek DSA.")
                        else:
                            logger.warning(f"Keeping user-set page_size={self.page_size} for DeepSeek DSA (EAGLE3 topk>1 compat).")'''

if old not in text:
    print("[ERROR] target not found"); sys.exit(1)

text = text.replace(old, new)
p.write_text(text)
print("[FIX] applied: DSA page_size override now respects user-set --page-size 1")
