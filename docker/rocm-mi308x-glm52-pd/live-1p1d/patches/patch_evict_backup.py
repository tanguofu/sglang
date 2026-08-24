#!/usr/bin/env python3
"""Patch unified_radix_cache.is_write_back: always defer write-back before eviction.

Root cause (PD prefill warm-cache miss for ~158K prompts):
  With hicache write_policy != "write_back" (write_through / write_through_selective),
  `is_write_back` is False, so `tree_core.evict_device_leaf` drops unbackuped
  GPU nodes entirely instead of returning a deferred BackupKV. The assumption
  is that write_through backs up on every cache hit, so an unbackuped node was
  never useful.

  PD prefill breaks this assumption: `cache_finished_req` inserts the node
  (GPU-resident, unbackuped). If it is evicted before its first hit (GPU pool
  pressure from a large ~158K insert), the KV is lost and the next identical
  request cold-prefills (~265s instead of ~16s).

  Verified empirically:
    - 158K prompt, 3s  gap between identical requests -> miss (292s)
    - 158K prompt, 30s gap between identical requests -> hit  (15.9s)

Fix:
  When HiCache is enabled (cache_controller is not None), set is_write_back
  True regardless of write policy, so eviction always runs the D->H backup
  before demoting instead of dropping.

Idempotent: matches the exact OLD block; skips if already patched.
"""
import sys

TARGET = (
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_radix_cache.py"
)
if len(sys.argv) > 1:
    TARGET = sys.argv[1]

OLD = """self.is_write_back = (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        )"""

NEW = """self.is_write_back = (
            # FIX(evict-backup): back up unbackuped nodes to host before device
            # eviction whenever HiCache is enabled, regardless of write policy.
            # Otherwise write_through eviction drops nodes inserted by PD
            # prefill (cache_finished_req) before their first hit, and the next
            # identical long-prompt request cold-prefills.
            self.cache_controller is not None
        )"""

with open(TARGET) as f:
    src = f.read()

count = src.count(OLD)
if count == 0:
    if "FIX(evict-backup)" in src:
        print("patch_evict_backup: already patched, skipping")
        sys.exit(0)
    raise RuntimeError(
        "patch_evict_backup: anchor not found — source layout changed?"
    )

src = src.replace(OLD, NEW, 1)
with open(TARGET, "w") as f:
    f.write(src)

with open(TARGET) as f:
    verify = f.read()
if verify.count(OLD) != 0 or "FIX(evict-backup)" not in verify:
    raise RuntimeError("patch_evict_backup: verification failed")
print("patch_evict_backup: SUCCESS")
