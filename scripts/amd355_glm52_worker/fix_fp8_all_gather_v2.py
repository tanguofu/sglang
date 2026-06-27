#!/usr/bin/env python3
"""Fix FP8 all_gather bug v2: cast FP8→bf16 before all_gather, then cast back to FP8 after.
NCCL doesn't support torch.float8_e4m3fn, so we cast to bf16 for the all_gather operation,
then cast back to the original FP8 dtype for attention (which expects FP8 activations to match FP8 weights)."""
import pathlib, sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/communicator.py"
p = pathlib.Path(FILE)
if not p.exists():
    print(f"[ERROR] file not found: {FILE}"); sys.exit(1)

text = p.read_text()

if "# FP8 all_gather fix v2" in text:
    print("[FIX] already patched v2"); sys.exit(0)

# Remove v1 fix if present
v1_marker = "# FP8 all_gather fix: NCCL doesn't support float8_e4m3fn, cast to bf16"
if v1_marker in text:
    # Find and remove v1 fix blocks
    import re
    # Remove v1 tuple path
    text = re.sub(r'\s*# FP8 all_gather fix:.*?\n.*?_orig_dtype.*?\n.*?if _orig_dtype.*?\n.*?local_hidden_states = local_hidden_states\.to\(torch\.bfloat16\)\n.*?output = output\.to\(torch\.bfloat16\)\n', '', text, flags=re.DOTALL)
    # Remove v1 non-tuple path  
    text = re.sub(r'\s*# FP8 all_gather fix: NCCL doesn\'t support.*?\n.*?if local_hidden_states\.dtype == torch\.float8_e4m3fn:\n.*?local_hidden_states = local_hidden_states\.to\(torch\.bfloat16\)\n.*?hidden_states = hidden_states\.to\(torch\.bfloat16\)\n', '', text, flags=re.DOTALL)
    print("[FIX] removed v1 fix")

# v2 fix: cast FP8→bf16 before all_gather, cast back to FP8 after
# Tuple path
old1 = """                attn_tp_all_gather_into_tensor(
                    output,
                    local_hidden_states,
                )
                gathered_hidden_states.append(output)"""
new1 = """                # FP8 all_gather fix v2: NCCL doesn't support float8_e4m3fn
                _orig_dtype = local_hidden_states.dtype
                if _orig_dtype == torch.float8_e4m3fn:
                    local_hidden_states = local_hidden_states.to(torch.bfloat16)
                    output = output.to(torch.bfloat16)
                attn_tp_all_gather_into_tensor(
                    output,
                    local_hidden_states,
                )
                if _orig_dtype == torch.float8_e4m3fn:
                    output = output.to(_orig_dtype)
                gathered_hidden_states.append(output)"""

# Non-tuple path
old2 = """        attn_tp_all_gather_into_tensor(
            hidden_states,
            local_hidden_states,
        )
        return hidden_states


class CommunicateWithAllReduceAndLayerNormFn:"""
new2 = """        # FP8 all_gather fix v2: NCCL doesn't support float8_e4m3fn
        _orig_dtype = local_hidden_states.dtype
        if _orig_dtype == torch.float8_e4m3fn:
            local_hidden_states = local_hidden_states.to(torch.bfloat16)
            hidden_states = hidden_states.to(torch.bfloat16)
        attn_tp_all_gather_into_tensor(
            hidden_states,
            local_hidden_states,
        )
        if _orig_dtype == torch.float8_e4m3fn:
            hidden_states = hidden_states.to(_orig_dtype)
        return hidden_states


class CommunicateWithAllReduceAndLayerNormFn:"""

if old1 not in text:
    print("[ERROR] tuple path target not found"); sys.exit(1)
if old2 not in text:
    print("[ERROR] non-tuple path target not found"); sys.exit(1)

text = text.replace(old1, new1)
text = text.replace(old2, new2)
p.write_text(text)
print("[FIX] applied v2: FP8→bf16 before all_gather, cast back to FP8 after")
