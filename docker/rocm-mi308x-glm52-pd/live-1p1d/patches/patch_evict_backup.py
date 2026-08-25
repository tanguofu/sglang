#!/usr/bin/env python3
"""v2: PD eviction D->H backup without disabling write_through L2/L3.

v1 set ``is_write_back = True`` whenever HiCache was enabled. That made
``tree_core`` skip ``hit_count`` and never ``write_backup`` on a cache hit,
so ``write_through_selective`` never promoted GPU KV to host (L2) or
Mooncake (L3). Cross-prefill prefix cache stayed empty.

v2 keeps ``is_write_back`` equal to the real write policy (False for
``write_through`` / ``write_through_selective``) so the 2nd hit still
promotes L2 then L3. Only the eviction call site forces D->H when HiCache
is on, which is the original PD bugfix:

  cache_finished_req inserts a GPU-resident unbackuped node. Under GPU
  pool pressure a large (~158K) insert can evict it before the 2nd hit.
  write_through eviction would drop that node; forcing backup on eviction
  keeps the KV on host so the next identical request can load back.

Idempotent across: fresh image, v1-patched tree, already-v2 tree.
"""
import sys

TARGET = (
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_radix_cache.py"
)
if len(sys.argv) > 1:
    TARGET = sys.argv[1]

IS_WRITE_BACK_POLICY = """self.is_write_back = (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        )"""

IS_WRITE_BACK_V1 = """self.is_write_back = (
            # FIX(evict-backup): back up unbackuped nodes to host before device
            # eviction whenever HiCache is enabled, regardless of write policy.
            # Otherwise write_through eviction drops nodes inserted by PD
            # prefill (cache_finished_req) before their first hit, and the next
            # identical long-prompt request cold-prefills.
            self.cache_controller is not None
        )"""

EVICT_OLD = (
    "result = self.tree_core.evict_device_leaf(node_id, self.is_write_back)"
)
EVICT_NEW = """result = self.tree_core.evict_device_leaf(
            node_id,
            # FIX(evict-backup-v2): PD prefill may evict unbackuped GPU nodes
            # before write_through_selective reaches threshold. Force D->H on
            # eviction whenever HiCache is on, but keep is_write_back tied to
            # the write policy so hits still promote L2/L3.
            self.is_write_back or self.cache_controller is not None,
        )"""


def main() -> None:
    with open(TARGET) as f:
        src = f.read()

    if "FIX(evict-backup-v2)" in src and IS_WRITE_BACK_POLICY in src:
        print("patch_evict_backup: v2 already applied, skipping")
        sys.exit(0)

    if IS_WRITE_BACK_V1 in src:
        src = src.replace(IS_WRITE_BACK_V1, IS_WRITE_BACK_POLICY, 1)
        print("patch_evict_backup: reverted v1 is_write_back override")
    elif IS_WRITE_BACK_POLICY not in src:
        raise RuntimeError(
            "patch_evict_backup: is_write_back policy assignment not found"
        )

    if "FIX(evict-backup-v2)" in src:
        pass
    elif src.count(EVICT_OLD) == 1:
        src = src.replace(EVICT_OLD, EVICT_NEW, 1)
    else:
        raise RuntimeError(
            "patch_evict_backup: evict_device_leaf call site not found "
            f"(count={src.count(EVICT_OLD)})"
        )

    with open(TARGET, "w") as f:
        f.write(src)

    with open(TARGET) as f:
        verify = f.read()
    if IS_WRITE_BACK_V1 in verify:
        raise RuntimeError("patch_evict_backup: v1 override still present")
    if IS_WRITE_BACK_POLICY not in verify:
        raise RuntimeError("patch_evict_backup: policy is_write_back missing")
    if "FIX(evict-backup-v2)" not in verify:
        raise RuntimeError("patch_evict_backup: v2 eviction site missing")
    if verify.count(EVICT_OLD) != 0:
        raise RuntimeError("patch_evict_backup: unpatched evict call remains")
    print("patch_evict_backup: SUCCESS v2")


if __name__ == "__main__":
    main()
