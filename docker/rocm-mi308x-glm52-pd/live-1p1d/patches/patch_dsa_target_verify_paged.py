#!/usr/bin/env python3
"""
Patch dsa_indexer.py: remove HIP-specific _get_topk_ragged branch for
target_verify that crashes during CUDA graph capture.

Root cause: During "Capture target verify CUDA graph", the metadata's
`indexer_k_start_end` is None, but `_get_topk_ragged` tries to unpack
`metadata.get_indexer_kvcache_range()` which returns None → TypeError.

Fix: Always use `_get_topk_paged` for target_verify (same as CUDA path),
which doesn't require `indexer_k_start_end`.

Idempotent.
"""
from pathlib import Path

TARGET = Path("/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py")

OLD = """                if _is_hip and forward_batch.forward_mode.is_target_verify():
                    topk_result = self._get_topk_ragged(
                        enable_dual_stream,
                        forward_batch,
                        layer_id,
                        q_fp8,
                        weights,
                        metadata,
                    )
                else:
                    topk_result = self._get_topk_paged(
                        forward_batch, layer_id, q_fp8, weights, metadata
                    )
"""
NEW = """                topk_result = self._get_topk_paged(
                    forward_batch, layer_id, q_fp8, weights, metadata
                )
"""

text = TARGET.read_text()
old_count = text.count(OLD)
new_count = text.count(NEW)

if old_count == 1 and new_count == 0:
    text = text.replace(OLD, NEW, 1)
    TARGET.write_text(text)
    status = "applied"
elif old_count == 0 and new_count == 1:
    status = "already-patched"
else:
    raise RuntimeError(f"Unexpected patch state: old={old_count}, new={new_count}")

# Verify
verified = TARGET.read_text()
if verified.count(NEW) != 1 or verified.count(OLD) != 0:
    raise RuntimeError("Verification failed")
print(f"DSA_TARGET_VERIFY_PAGED_PATCH={status}")
