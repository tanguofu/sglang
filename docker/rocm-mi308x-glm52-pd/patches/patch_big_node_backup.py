#!/usr/bin/env python3
"""v1: large-node immediate D->H backup on first insert (agent-context fix).

Problem (2026-09-07 bench, salt 1788769057):
  write_through_selective threshold=2 means a prefix node is only backed up to
  host (L2) on its SECOND insert. A codex agent loop is exactly
  "turn1 cold insert -> turn2 warm hit -> turn3 ...": turn1's 200K blob sits
  unbackuped in GPU radix. Between turn2's finish (backup fires, async) and
  turn3's arrival 5s later, the blob became unmatchable (cached=0, full 81s
  re-prefill). A 30-min-later rerun of the same blob hit 196224/196224 from
  GPU radix, so the cache itself works — the miss is the threshold window.

Fix: nodes larger than SGLANG_HICACHE_BIG_NODE_TOKENS (default 65536) skip the
hit-count threshold and back up on the FIRST insert. Small chat prefixes keep
the selective policy (threshold=2), so D->H DMA volume stays near current
levels — only the big agent-context nodes (the ones whose re-prefill costs
80s) get the immediate backup.

Idempotent: safe to re-run on a fresh image or an already-patched tree.
"""
import sys

TARGET = (
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_cache/"
    "unified_tree_core.py"
)
if len(sys.argv) > 1:
    TARGET = sys.argv[1]

OLD = '''    def _inc_hit_count_and_check(
        self, node: UnifiedTreeNode, chunked: bool = False
    ) -> bool:
        """Increment hit count; check whether a write backup should be fired."""
        if node.evicted or chunked:
            return False
        if self.is_write_back:
            return False
        node.hit_count += 1
        return (
            self.enable_hicache
            and not node.backuped
            and node.hit_count >= self.write_through_threshold
        )
'''

NEW = '''    def _inc_hit_count_and_check(
        self, node: UnifiedTreeNode, chunked: bool = False
    ) -> bool:
        """Increment hit count; check whether a write backup should be fired."""
        if node.evicted or chunked:
            return False
        if self.is_write_back:
            return False
        node.hit_count += 1
        if not (self.enable_hicache and not node.backuped):
            return False
        # FIX(big-node-backup): agent-context nodes (> threshold tokens) back
        # up on the FIRST insert. write_through_selective's hit-count gate
        # leaves turn-1's large prefix unbackuped; a turn-3 arrival inside the
        # async backup window then cold-prefills 80s. Big nodes only — small
        # chat prefixes keep the selective policy.
        from sglang.srt.utils.common import get_int_env_var

        big_node_tokens = get_int_env_var("SGLANG_HICACHE_BIG_NODE_TOKENS", 65536)
        if big_node_tokens > 0 and len(node.key) >= big_node_tokens:
            return True
        return node.hit_count >= self.write_through_threshold
'''


def main() -> None:
    with open(TARGET) as f:
        src = f.read()

    if "FIX(big-node-backup)" in src:
        print("patch_big_node_backup: already applied, skipping")
        sys.exit(0)

    if src.count(OLD) != 1:
        raise RuntimeError(
            f"patch_big_node_backup: anchor not found exactly once "
            f"(count={src.count(OLD)})"
        )

    src = src.replace(OLD, NEW, 1)

    with open(TARGET, "w") as f:
        f.write(src)

    with open(TARGET) as f:
        verify = f.read()
    if "FIX(big-node-backup)" not in verify:
        raise RuntimeError("patch_big_node_backup: patch missing after write")
    if verify.count(OLD) != 0:
        raise RuntimeError("patch_big_node_backup: unpatched anchor remains")
    print("patch_big_node_backup: SUCCESS")


if __name__ == "__main__":
    main()
