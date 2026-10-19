#!/usr/bin/env python3
"""Prefer FlyDSL gfx942 fp8 MQA logits; Triton remains the ImportError fallback.

Do not COPY python/sglang/.../dsa_indexer.py over the 0.5.17 image tree — local
indexer has diverged (2490 vs 1863 lines) and would drop fused-store-length-guard.
This patch only swaps the two HIP import sites after post1 dsa_indexer.py runs.
"""
from pathlib import Path

TARGET = Path("/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py")

OLD = "                    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits\n"
NEW = (
    "                    # WAVE1: FlyDSL gfx942 MQA logits (Triton fallback)\n"
    "                    try:\n"
    "                        from aiter.ops.flydsl.kernels.fp8_mqa_logits import (\n"
    "                            flydsl_fp8_mqa_logits as fp8_mqa_logits,\n"
    "                        )\n"
    "                    except ImportError:\n"
    "                        from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits\n"
)

text = TARGET.read_text()
if "flydsl_fp8_mqa_logits" in text:
    status = "already-patched"
elif text.count(OLD) == 2:
    text = text.replace(OLD, NEW)
    TARGET.write_text(text)
    status = "applied"
else:
    raise RuntimeError(
        f"flydsl MQA patch: expected 2 Triton HIP imports, found {text.count(OLD)}"
    )

verified = TARGET.read_text()
if "flydsl_fp8_mqa_logits" not in verified:
    raise RuntimeError("flydsl MQA patch verification failed: import missing")
if verified.count("WAVE1: FlyDSL gfx942 MQA logits") != 2:
    raise RuntimeError("flydsl MQA patch verification failed: expected 2 WAVE1 sites")
print(f"FLYDSL_MQA_PATCH={status}")
