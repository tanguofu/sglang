#!/usr/bin/env python3
"""Log L3 prefetch exist hits at INFO so cross-P misses are visible.

Also upgrades the revoke path from debug to info. Idempotent.
"""
import sys

CC = "/sgl-workspace/sglang/python/sglang/srt/managers/cache_controller.py"
HIRADIX = "/sgl-workspace/sglang/python/sglang/srt/mem_cache/hiradix_cache.py"

OLD_RETURN = """        return hash_value, storage_query_count
"""

NEW_RETURN = """        # FIX(prefetch-log): INFO so we can see exist-hit vs GET on PD.
        logger.info(
            "L3 storage_hit_query tokens=%s pages=%s last_hash=%s first=%s",
            storage_query_count,
            len(hash_value),
            last_hash,
            (hash_value[0][:16] if hash_value else None),
        )
        return hash_value, storage_query_count
"""

OLD_REVOKE = """                    logger.debug(
                        f"Revoking prefetch for request {req_id} due to insufficient hits ({operation.storage_hit_count})."
                    )
"""

NEW_REVOKE = """                    logger.info(
                        # FIX(prefetch-log)
                        f"Revoking prefetch for request {req_id} due to insufficient hits ({operation.storage_hit_count})."
                    )
"""


def patch_file(path: str, old: str, new: str, marker: str) -> None:
    with open(path) as f:
        src = f.read()
    if marker in src and old not in src:
        print(f"patch_prefetch_log: {path} already patched, skipping")
        return
    if old not in src:
        raise RuntimeError(f"patch_prefetch_log: pattern not found in {path}")
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print(f"patch_prefetch_log: patched {path}")


def main() -> None:
    if len(sys.argv) > 1:
        global CC, HIRADIX
        # unused; keep signature stable
    patch_file(CC, OLD_RETURN, NEW_RETURN, "FIX(prefetch-log)")
    patch_file(HIRADIX, OLD_REVOKE, NEW_REVOKE, "FIX(prefetch-log)")
    print("patch_prefetch_log: SUCCESS")


if __name__ == "__main__":
    main()
