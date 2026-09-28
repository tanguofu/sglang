#!/usr/bin/env python3
"""Append GLM-5.2 decode MoE tiles (expert=256, topk=8) to the host FMOE CSV.

Prefill hits expert=257,topk=9 (shared-expert fusion) and already has named
1stage kernels. Decode EAGLE/graph capture hits expert=256,topk=8; only
token=4,8 are in the live CSV, so token=1,2,16,32 fall back to 2stage default
and 64,128 to 1stage default.

Clone err1=0.0% 257/9 tiles onto 256/8 for decode graph sizes. Dedup by the
aiter primary key. Keep every 257/9 row.

Usage (inside a worker pod; /data is hostPath):
  python3 merge_decode_fmoe_256_8.py
  python3 merge_decode_fmoe_256_8.py /data/aiter_configs/a8w8_blockscale_tuned_fmoe_glm5_1_cu80.csv
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

DEFAULT = Path("/data/aiter_configs/a8w8_blockscale_tuned_fmoe_glm5_1_cu80.csv")
DECODE_TOKENS = {1, 2, 4, 8, 16, 32, 64, 128}
PRIMARY = (
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
)


def key_of(row: dict) -> tuple:
    return tuple(row.get(c, "") for c in PRIMARY)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    if not path.is_file():
        print(f"missing {path}", file=sys.stderr)
        return 1

    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]

    kept = []
    seen = set()
    dups = 0
    for row in rows:
        k = key_of(row)
        if k in seen:
            dups += 1
            continue
        seen.add(k)
        kept.append(row)

    cloned = 0
    skipped_existing = 0
    skipped_err = 0
    for row in list(kept):
        if row.get("expert") != "257" or row.get("topk") != "9":
            continue
        try:
            token = int(row.get("token", ""))
        except ValueError:
            continue
        if token not in DECODE_TOKENS:
            continue
        if row.get("err1") not in ("0.0%", "0%", "0.0", "0"):
            skipped_err += 1
            continue
        clone = dict(row)
        clone["expert"] = "256"
        clone["topk"] = "8"
        ck = key_of(clone)
        if ck in seen:
            skipped_existing += 1
            continue
        seen.add(ck)
        kept.append(clone)
        cloned += 1

    bak = path.with_suffix(path.suffix + ".bak-wave1")
    if not bak.exists():
        shutil.copy2(path, bak)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    tmp.replace(path)

    tokens_256 = sorted(
        int(r["token"])
        for r in kept
        if r.get("expert") == "256" and r.get("topk") == "8" and str(r.get("token", "")).isdigit()
    )
    tokens_257 = sorted(
        int(r["token"])
        for r in kept
        if r.get("expert") == "257" and r.get("topk") == "9" and str(r.get("token", "")).isdigit()
    )
    print(
        f"wrote {path} rows={len(kept)} dropped_dups={dups} cloned_256_8={cloned} "
        f"skipped_existing={skipped_existing} skipped_err={skipped_err} "
        f"tokens_256_8={tokens_256} tokens_257_9={tokens_257}"
    )
    missing = sorted(DECODE_TOKENS - set(tokens_256))
    if missing:
        print(f"WARNING missing decode tokens {missing}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
