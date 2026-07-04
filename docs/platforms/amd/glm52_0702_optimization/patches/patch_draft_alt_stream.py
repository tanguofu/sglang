#!/usr/bin/env python3
"""Fix: DeepseekNextnModel (draft model) doesn't check SGLANG_ROCM_USE_MULTI_STREAM
when creating alt_stream. The target model (DeepseekV2Model) does check it.

Root cause: deepseek_nextn.py:126 missing SGLANG_ROCM_USE_MULTI_STREAM check.
Note: _is_musa is NOT imported in deepseek_nextn.py, so we only use _is_cuda.

Also add defensive null check in _pre_combine_hook (deepseek_v2.py:1080).
"""
import sys

NEXTN = "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py"
DSV2 = "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py"

# --- Fix 1: deepseek_nextn.py — add SGLANG_ROCM_USE_MULTI_STREAM to alt_stream ---
with open(NEXTN, "r") as f:
    nextn_content = f.read()

old_nextn = """        self.alt_stream = (
            torch.cuda.Stream()
            if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            else None
        )"""

new_nextn = """        self.alt_stream = (
            torch.cuda.Stream()
            if (
                _is_cuda
                or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
                or envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
            )
            else None
        )"""

if old_nextn in nextn_content:
    nextn_content = nextn_content.replace(old_nextn, new_nextn, 1)
    with open(NEXTN, "w") as f:
        f.write(nextn_content)
    print("[OK] Patched deepseek_nextn.py: alt_stream now checks SGLANG_ROCM_USE_MULTI_STREAM")
elif "SGLANG_ROCM_USE_MULTI_STREAM" in nextn_content:
    print("[SKIP] deepseek_nextn.py already has SGLANG_ROCM_USE_MULTI_STREAM")
else:
    print("[ERROR] Cannot find alt_stream pattern in deepseek_nextn.py")
    sys.exit(1)

# --- Fix 2: deepseek_v2.py:1080 — defensive null check in _pre_combine_hook ---
with open(DSV2, "r") as f:
    dsv2_content = f.read()

old_hook_full = """            def _pre_combine_hook(dispatcher, combine_input):
                nonlocal shared_output
                self.alt_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.alt_stream):
                    shared_output = self._forward_shared_experts(
                        hidden_states_for_shared_experts
                    )
                torch.cuda.current_stream().wait_stream(self.alt_stream)"""

new_hook_full = """            def _pre_combine_hook(dispatcher, combine_input):
                nonlocal shared_output
                if self.alt_stream is not None:
                    self.alt_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.alt_stream):
                        shared_output = self._forward_shared_experts(
                            hidden_states_for_shared_experts
                        )
                    torch.cuda.current_stream().wait_stream(self.alt_stream)
                else:
                    shared_output = self._forward_shared_experts(
                        hidden_states_for_shared_experts
                    )"""

if old_hook_full in dsv2_content:
    dsv2_content = dsv2_content.replace(old_hook_full, new_hook_full, 1)
    with open(DSV2, "w") as f:
        f.write(dsv2_content)
    print("[OK] Patched deepseek_v2.py: _pre_combine_hook null check for alt_stream")
elif new_hook_full in dsv2_content:
    print("[SKIP] deepseek_v2.py _pre_combine_hook already patched")
else:
    print("[INFO] deepseek_v2.py _pre_combine_hook pattern not found (may already be guarded)")

print("\n=== Draft alt_stream + _pre_combine_hook null check patch complete ===")
