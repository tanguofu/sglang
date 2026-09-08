#!/usr/bin/env python3
"""Fix: chunked-insert FreeDeviceKV freeing tree-owned slots (zombie cache).

Root cause (2026-09-07, GLM-5.3 2P2D canary):
  A chunked prefill's insert walk can match MORE of the radix tree than
  the request's cache_protected_len (prev_prefix_len) — another
  request's insert extended the shared prefix path between this
  chunk's schedule and its insert. In that region the request's
  req_to_token row aliases the TREE nodes' slots. The duplicate-free
  computed dup_start = max(0, prev_prefix_len - total_prefix_length)
  clamps to 0 there, and FreeDeviceKV([value_slice[0:consumed_from]])
  frees the TREE nodes' own slots while the nodes stay in the tree.

  Result: zombie cache nodes (tree refs freed slots). The next request
  either "hits" stale slots (fast follow-up) or misses after the slots
  are reused (slow follow-up) — the non-deterministic turn-3 miss
  (200K blob cold-prefilling 81s after a warm turn-2).

Fix: when the walk has reached or passed prev_prefix_len
  (total_prefix_length >= prev_prefix_len), the node's span lies
  entirely in the tree-matched region — the request never owned those
  slots, so there is nothing to free. Skip the FreeDeviceKV.

This mirrors RadixCache.cache_unfinished_req's free range
  [cache_protected_len : new_prefix_len] — empty when the walk never
  got ahead of the protected boundary.

Idempotent.
"""
import sys

TARGET = (
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_cache/"
    "unified_tree_core.py"
)
if len(sys.argv) > 1:
    TARGET = sys.argv[1]

OLD = """            dup_start = max(0, state.params.prev_prefix_len - state.total_prefix_length)
            if dup_start < consumed_from:
                step_actions.append(
                    FreeDeviceKV([value_slice[dup_start:consumed_from]])
                )
"""

NEW = """            # FIX(zombie-free): the walk can match beyond prev_prefix_len
            # (another request extended the shared prefix path between this
            # chunk's schedule and insert). Past that boundary the request's
            # req_to_token row aliases the TREE nodes' slots — freeing them
            # zombifies the tree nodes. Only free within the request-owned
            # region, which starts strictly before prev_prefix_len.
            if state.params.prev_prefix_len > state.total_prefix_length:
                dup_start = (
                    state.params.prev_prefix_len - state.total_prefix_length
                )
                if dup_start < consumed_from:
                    step_actions.append(
                        FreeDeviceKV([value_slice[dup_start:consumed_from]])
                    )
"""


def main() -> None:
    with open(TARGET) as f:
        src = f.read()

    if "FIX(zombie-free)" in src:
        print("patch_zombie_free_fix: already applied, skipping")
        sys.exit(0)

    if src.count(OLD) != 1:
        raise RuntimeError(
            f"patch_zombie_free_fix: anchor count={src.count(OLD)}"
        )

    src = src.replace(OLD, NEW, 1)

    with open(TARGET, "w") as f:
        f.write(src)

    with open(TARGET) as f:
        verify = f.read()
    if "FIX(zombie-free)" not in verify:
        raise RuntimeError("patch_zombie_free_fix: patch missing after write")
    if verify.count(OLD) != 0:
        raise RuntimeError("patch_zombie_free_fix: unpatched anchor remains")
    print("patch_zombie_free_fix: SUCCESS")


if __name__ == "__main__":
    main()
