#!/usr/bin/env python3
"""Fix FP8 all_gather bug: NCCL doesn't support torch.float8_e4m3fn dtype.
Cast FP8 hidden_states to bf16 before attn_tp_all_gather_into_tensor,
since NCCL cannot all_gather FP8 tensors natively.

Root cause: communicator.py:_scattered_to_tp_attn_full calls
  attn_tp_all_gather_into_tensor(output, local_hidden_states)
where local_hidden_states may be torch.float8_e4m3fn (from fused_rms_fp8_group_quant
on gfx95+aiter+fp8 path). NCCL's ncclDataTypeEnum.from_torch doesn't support fp8.

Fix: cast FP8 → bf16 before all_gather. Output stays bf16 (attention handles bf16).
"""
import pathlib, sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/communicator.py"
p = pathlib.Path(FILE)
if not p.exists():
    print(f"[ERROR] file not found: {FILE}"); sys.exit(1)

text = p.read_text()

if "# FP8 all_gather fix" in text:
    print("[FIX] already patched"); sys.exit(0)

# Fix 1: tuple path (line ~902)
old1 = """                attn_tp_all_gather_into_tensor(
                    output,
                    local_hidden_states,
                )
                gathered_hidden_states.append(output)"""
new1 = """                # FP8 all_gather fix: NCCL doesn't support float8_e4m3fn, cast to bf16
                _orig_dtype = local_hidden_states.dtype
                if _orig_dtype == torch.float8_e4m3fn:
                    local_hidden_states = local_hidden_states.to(torch.bfloat16)
                    output = output.to(torch.bfloat16)
                attn_tp_all_gather_into_tensor(
                    output,
                    local_hidden_states,
                )
                gathered_hidden_states.append(output)"""

# Fix 2: non-tuple path (line ~913)
old2 = """        attn_tp_all_gather_into_tensor(
            hidden_states,
            local_hidden_states,
        )
        return hidden_states


class CommunicateWithAllReduceAndLayerNormFn:"""
new2 = """        # FP8 all_gather fix: NCCL doesn't support float8_e4m3fn, cast to bf16
        if local_hidden_states.dtype == torch.float8_e4m3fn:
            local_hidden_states = local_hidden_states.to(torch.bfloat16)
            hidden_states = hidden_states.to(torch.bfloat16)
        attn_tp_all_gather_into_tensor(
            hidden_states,
            local_hidden_states,
        )
        return hidden_states


class CommunicateWithAllReduceAndLayerNormFn:"""

if old1 not in text:
    print("[ERROR] tuple path target not found"); sys.exit(1)
if old2 not in text:
    print("[ERROR] non-tuple path target not found"); sys.exit(1)

text = text.replace(old1, new1)
text = text.replace(old2, new2)
p.write_text(text)
print("[FIX] applied: FP8 all_gather cast to bf16 in _scattered_to_tp_attn_full (both tuple and non-tuple paths)")
