#!/usr/bin/env python3
"""Port patch 03 (elementwise q_fp8_raw uint8-then-view on HIP) to post1.

Semantic change: on HIP, allocating a tensor directly as float8_e4m3fn and passing it
to the sgl_kernel op triggers an FN/FNUZ dtype mismatch. Allocate as uint8, pass uint8
to the kernel, then view-as float8_e4m3fn for the return value. CUDA keeps the direct
float8 allocation.

Applied to BOTH fused_q_indexer_rope_hadamard_quant and fused_q_indexer_rope_first_quant.

NOTE: the branch version had a latent bug (allocated q_fp8_raw but passed `q_fp8` to the
HIP kernel path). This port passes q_fp8_raw consistently and is correct.

Idempotent.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/port-post1/base/jit_kernel_dsv4_elementwise.py")
src = path.read_text()
changed = []

# ---- Function 1: fused_q_indexer_rope_hadamard_quant ----
# Replace the direct float8 allocation + add the view at the end.
f1_alloc_old = "    q_fp8 = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)\n    weights_out = torch.empty("
f1_alloc_new = (
    "    if _is_hip:\n"
    "        q_fp8_raw = torch.empty(q_input.shape, dtype=torch.uint8, device=q_input.device)\n"
    "    else:\n"
    "        q_fp8_raw = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)\n"
    "    weights_out = torch.empty("
)
# In f1, the kernel calls (HIP + CUDA branches) pass `q_fp8` -> change to q_fp8_raw.
# And the return: `return q_fp8, weights_out` -> view q_fp8_raw first.
if "q_fp8_raw.view(torch.float8_e4m3fn) if _is_hip else q_fp8_raw" in src:
    changed.append("f1: skipped")
else:
    assert f1_alloc_old in src, "03 f1: alloc anchor not found"
    src = src.replace(f1_alloc_old, f1_alloc_new, 1)
    # The HIP sgl_kernel call passes q_fp8 -> q_fp8_raw
    f1_hip_call_old = (
        "        torch.ops.sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant(\n"
        "            q_input,\n"
        "            q_fp8,\n"
        "            weight,\n"
        "            weights_out,\n"
        "            float(weight_scale),\n"
        "            freqs_real,\n"
        "            positions,\n"
        "        )"
    )
    f1_hip_call_new = f1_hip_call_old.replace("            q_fp8,\n", "            q_fp8_raw,\n", 1)
    assert f1_hip_call_old in src, "03 f1: hip kernel call not found"
    src = src.replace(f1_hip_call_old, f1_hip_call_new, 1)
    # CUDA module.forward also passes q_fp8 -> q_fp8_raw
    f1_cuda_call_old = (
        "        module.forward(\n"
        "            q_input,\n"
        "            q_fp8,\n"
        "            weight,\n"
        "            weights_out,\n"
        "            float(weight_scale),\n"
        "            freqs_real,\n"
        "            positions,\n"
        "        )"
    )
    f1_cuda_call_new = f1_cuda_call_old.replace("            q_fp8,\n", "            q_fp8_raw,\n", 1)
    assert f1_cuda_call_old in src, "03 f1: cuda module call not found"
    src = src.replace(f1_cuda_call_old, f1_cuda_call_new, 1)
    # xpu branch also passes q_fp8 -> q_fp8_raw
    f1_xpu_call_old = (
        "        fused_q_indexer_rope_hadamard_quant(\n"
        "            q_input,\n"
        "            q_fp8,\n"
        "            weight,\n"
        "            weights_out,\n"
        "            float(weight_scale),\n"
        "            freqs_real,\n"
        "            positions,\n"
        "        )"
    )
    if f1_xpu_call_old in src:
        src = src.replace(f1_xpu_call_old, f1_xpu_call_old.replace("            q_fp8,\n", "            q_fp8_raw,\n", 1), 1)
    # return: add view
    f1_ret_old = "    return q_fp8, weights_out\n\n\ndef fused_q_indexer_rope_first_quant"
    f1_ret_new = "    q_fp8 = q_fp8_raw.view(torch.float8_e4m3fn) if _is_hip else q_fp8_raw\n    return q_fp8, weights_out\n\n\ndef fused_q_indexer_rope_first_quant"
    assert f1_ret_old in src, "03 f1: return anchor not found"
    src = src.replace(f1_ret_old, f1_ret_new, 1)
    changed.append("f1: applied")

# ---- Function 2: fused_q_indexer_rope_first_quant ----
f2_alloc_old = "    q_fp8 = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)\n    weights_out = torch.empty(\n        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device\n    )\n    module = _jit_main_q_indexer_rope_first_quant_module(q_input.dtype)"
f2_alloc_new = (
    "    from sglang.srt.utils import is_hip as _check_hip\n"
    "    _is_hip_local = _check_hip()\n"
    "    if _is_hip_local:\n"
    "        q_fp8_raw = torch.empty(q_input.shape, dtype=torch.uint8, device=q_input.device)\n"
    "    else:\n"
    "        q_fp8_raw = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)\n"
    "    weights_out = torch.empty(\n"
    "        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device\n"
    "    )\n"
    "    module = _jit_main_q_indexer_rope_first_quant_module(q_input.dtype)"
)
if "_is_hip_local" in src:
    changed.append("f2: skipped")
else:
    assert f2_alloc_old in src, "03 f2: alloc anchor not found"
    src = src.replace(f2_alloc_old, f2_alloc_new, 1)
    # module.forward passes q_fp8 -> q_fp8_raw
    f2_call_old = (
        "    module.forward(\n"
        "        q_input,\n"
        "        q_fp8,\n"
        "        weight,\n"
        "        weights_out,\n"
        "        float(weight_scale),\n"
        "        cos_sin_cache,\n"
        "        positions,\n"
        "    )"
    )
    f2_call_new = (
        "    module.forward(\n"
        "        q_input,\n"
        "        q_fp8_raw,\n"
        "        weight,\n"
        "        weights_out,\n"
        "        float(weight_scale),\n"
        "        cos_sin_cache,\n"
        "        positions,\n"
        "    )"
    )
    assert f2_call_old in src, "03 f2: module.forward call not found"
    src = src.replace(f2_call_old, f2_call_new, 1)
    # return: add view
    f2_ret_old = "    return q_fp8, weights_out"
    # Only the LAST occurrence (func2 return). Replace last occurrence.
    idx = src.rfind(f2_ret_old)
    assert idx != -1, "03 f2: return anchor not found"
    f2_ret_new = "    q_fp8 = q_fp8_raw.view(torch.float8_e4m3fn) if _is_hip_local else q_fp8_raw\n    return q_fp8, weights_out"
    src = src[:idx] + src[idx:].replace(f2_ret_old, f2_ret_new, 1)
    changed.append("f2: applied")

path.write_text(src)
print(f"[ok] {path}: " + ", ".join(changed))
