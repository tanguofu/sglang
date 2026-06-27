#!/usr/bin/env python3
"""Root fix: NCCL doesn't support float8_e4m3fn dtype, but FP8 = 1 byte = uint8.
Add float8_e4m3fn → ncclUint8 mapping in from_torch().
This allows NCCL all_gather to transfer FP8 tensors as uint8 (lossless, no cast needed).
NCCL all_gather is just memcpy (no reduce), so uint8 transfer is correct for FP8 data.

Note: This only works for all_gather (copy), NOT for all_reduce/reduce (sum).
But EP8+MTP only uses all_gather (moe_tp=1, no all_reduce), so this is safe.
"""
import pathlib, sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py"
p = pathlib.Path(FILE)
if not p.exists():
    print(f"[ERROR] file not found: {FILE}"); sys.exit(1)

text = p.read_text()

if "float8_e4m3fn" in text and "ncclUint8" in text:
    # Check if already has the mapping
    if "torch.float8_e4m3fn" in text:
        print("[FIX] already patched"); sys.exit(0)

# Add FP8 → uint8 mapping in from_torch, right after the uint8 check
old = """        if dtype == torch.uint8:
            return cls.ncclUint8"""
new = """        if dtype == torch.uint8:
            return cls.ncclUint8
        # FP8 = 1 byte = uint8 for NCCL transfer (all_gather only, not reduce)
        if dtype == torch.float8_e4m3fn:
            return cls.ncclUint8
        if dtype == torch.float8_e5m2:
            return cls.ncclUint8"""

if old not in text:
    print("[ERROR] target not found"); sys.exit(1)

text = text.replace(old, new)
p.write_text(text)
print("[FIX] applied: FP8 → ncclUint8 mapping added in from_torch (lossless, no cast needed)")
