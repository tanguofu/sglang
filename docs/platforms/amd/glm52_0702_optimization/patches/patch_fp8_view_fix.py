#!/usr/bin/env python3
"""Fix FP8 dtype mismatch in JIT kernel on HIP/ROCm.

Works with both 0629 (freqs_cis) and 0702 (cos_sin_cache) code variants.
"""

ELEMENTWISE_DSV4 = "/sgl-workspace/sglang/python/sglang/jit_kernel/dsv4/elementwise.py"

with open(ELEMENTWISE_DSV4, "r") as f:
    content = f.read()

patches_applied = 0

# --- Fix 1: fused_q_indexer_rope_first_quant ---
# Variant A: 0629 code (uses freqs_cis + freqs_real)
OLD_A = '''    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    q_fp8 = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    module = _jit_main_q_indexer_rope_first_quant_module(q_input.dtype)
    module.forward(
        q_input,
        q_fp8,
        weight,
        weights_out,
        float(weight_scale),
        freqs_real,
        positions,
    )
    return q_fp8, weights_out'''

NEW_A = '''    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    if _is_hip:
        q_fp8_raw = torch.empty(q_input.shape, dtype=torch.uint8, device=q_input.device)
    else:
        q_fp8_raw = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    module = _jit_main_q_indexer_rope_first_quant_module(q_input.dtype)
    module.forward(
        q_input,
        q_fp8_raw,
        weight,
        weights_out,
        float(weight_scale),
        freqs_real,
        positions,
    )
    q_fp8 = q_fp8_raw.view(torch.float8_e4m3fn) if _is_hip else q_fp8_raw
    return q_fp8, weights_out'''

# Variant B: 0702 code (uses cos_sin_cache directly, no freqs_real)
OLD_B = '''    q_fp8 = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    module = _jit_main_q_indexer_rope_first_quant_module(q_input.dtype)
    module.forward(
        q_input,
        q_fp8,
        weight,
        weights_out,
        float(weight_scale),
        cos_sin_cache,
        positions,
    )
    return q_fp8, weights_out'''

NEW_B = '''    from sglang.srt.utils import is_hip as _check_hip
    _is_hip_local = _check_hip()
    if _is_hip_local:
        q_fp8_raw = torch.empty(q_input.shape, dtype=torch.uint8, device=q_input.device)
    else:
        q_fp8_raw = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    module = _jit_main_q_indexer_rope_first_quant_module(q_input.dtype)
    module.forward(
        q_input,
        q_fp8_raw,
        weight,
        weights_out,
        float(weight_scale),
        cos_sin_cache,
        positions,
    )
    q_fp8 = q_fp8_raw.view(torch.float8_e4m3fn) if _is_hip_local else q_fp8_raw
    return q_fp8, weights_out'''

# Check if already patched
if "q_fp8_raw" in content and "fused_q_indexer_rope_first_quant" in content:
    print("[SKIP] Already patched: fused_q_indexer_rope_first_quant FP8 view fix")
elif NEW_A in content:
    print("[SKIP] Already patched: fused_q_indexer_rope_first_quant FP8 view fix (variant A)")
elif NEW_B in content:
    print("[SKIP] Already patched: fused_q_indexer_rope_first_quant FP8 view fix (variant B)")
elif OLD_A in content:
    content = content.replace(OLD_A, NEW_A, 1)
    print("[OK] Patched: fused_q_indexer_rope_first_quant FP8 view fix (variant A - freqs_cis)")
    patches_applied += 1
elif OLD_B in content:
    content = content.replace(OLD_B, NEW_B, 1)
    print("[OK] Patched: fused_q_indexer_rope_first_quant FP8 view fix (variant B - cos_sin_cache)")
    patches_applied += 1
else:
    print("[WARN] Pattern not found: fused_q_indexer_rope_first_quant (checking for partial match)")
    if "fused_q_indexer_rope_first_quant" in content:
        idx = content.index("def fused_q_indexer_rope_first_quant")
        end = content.index("\ndef ", idx + 10) if "\ndef " in content[idx+10:] else len(content)
        for line in content[idx:end].split("\n")[:25]:
            print(f"     {line}")

# --- Fix 2: fused_q_indexer_rope_hadamard_quant (preventive) ---
OLD_HAD = '''    q_fp8 = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    if _is_hip:
        torch.ops.sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant('''

NEW_HAD = '''    if _is_hip:
        q_fp8_raw = torch.empty(q_input.shape, dtype=torch.uint8, device=q_input.device)
    else:
        q_fp8_raw = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    if _is_hip:
        torch.ops.sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant('''

if NEW_HAD in content:
    print("[SKIP] Already patched: fused_q_indexer_rope_hadamard_quant FP8 view fix")
elif OLD_HAD in content:
    content = content.replace(OLD_HAD, NEW_HAD, 1)
    OLD_RET = '''    return q_fp8, weights_out


def fused_q_indexer_rope_first_quant('''
    NEW_RET = '''    q_fp8 = q_fp8_raw.view(torch.float8_e4m3fn) if _is_hip else q_fp8_raw
    return q_fp8, weights_out


def fused_q_indexer_rope_first_quant('''
    if OLD_RET in content:
        content = content.replace(OLD_RET, NEW_RET, 1)
        print("[OK] Patched: fused_q_indexer_rope_hadamard_quant FP8 view fix + return")
        patches_applied += 1
    else:
        print("[WARN] Could not find return statement for hadamard_quant")
else:
    print("[WARN] Pattern not found: fused_q_indexer_rope_hadamard_quant")

with open(ELEMENTWISE_DSV4, "w") as f:
    f.write(content)

print(f"\n=== FP8 view fix complete ({patches_applied} patches applied) ===")
