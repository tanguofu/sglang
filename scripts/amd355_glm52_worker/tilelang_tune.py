#!/usr/bin/env python3
"""Patch tilelang_kernel.py gfx950 tuning parameters."""
import sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/tilelang_kernel.py"

# Get params from command line
block_I = int(sys.argv[1]) if len(sys.argv) > 1 else 64
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 512
num_stages = int(sys.argv[3]) if len(sys.argv) > 3 else 0
block_per_cu = int(sys.argv[4]) if len(sys.argv) > 4 else 2

with open(FILE) as f:
    content = f.read()

old = "block_I, threads, num_stages, block_per_cu, cu = 64, 512, 0, 2, 256"
new = f"block_I, threads, num_stages, block_per_cu, cu = {block_I}, {threads}, {num_stages}, {block_per_cu}, 256"

if old in content:
    content = content.replace(old, new)
    with open(FILE, "w") as f:
        f.write(content)
    print(f"PATCHED: block_I={block_I}, threads={threads}, num_stages={num_stages}, block_per_cu={block_per_cu}")
else:
    # Maybe already patched — try to find the current line
    import re
    pattern = r"block_I, threads, num_stages, block_per_cu, cu = \d+, \d+, \d+, \d+, 256"
    if re.search(pattern, content):
        content = re.sub(pattern, new, content)
        with open(FILE, "w") as f:
            f.write(content)
        print(f"REPATCHED: block_I={block_I}, threads={threads}, num_stages={num_stages}, block_per_cu={block_per_cu}")
    else:
        print("ERROR: could not find target line")
        sys.exit(1)
