#!/usr/bin/env python3
"""Generate a thin MI308X BF16 GEMM overlay and overwrite the image file.

This is NOT a merge of /data/aiter_configs/bf16_tuned_gemm.csv or glm5 dummy
rows. It builds one standalone table for GLM-5.2 gate + DSA indexer:

  K=6144  N=32   indexer weights_proj (kept BF16 in the FP8 checkpoint)
  K=6144  N=256  MoE router gate     (kept BF16 in the FP8 checkpoint)

Existing image overlay rows for other N (e.g. N=160) are passed through so
we do not regress older MI308X tunes.

Locatable artifacts (hostPath /data, one copy per GPU node):

  /data/aiter_configs/gen_bf16_gate_indexer.py          this script
  /data/aiter_configs/mi308x_bf16_gate_indexer.csv      the table
  /data/aiter_configs/mi308x_bf16_gate_indexer.meta     version + counts

InitContainer:  python3 ... generate
Main container: python3 ... install   # wholesale overwrite of the image overlay

Usage:
  python3 gen_bf16_gate_indexer.py generate --workers 8
  python3 gen_bf16_gate_indexer.py install
  python3 gen_bf16_gate_indexer.py tune --workers 8   # optional GPU retune
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

VERSION = "1"
GFX = "gfx942"
CU = "80"
K = 6144
N_GENERATE = (32, 256)
N_PASSTHROUGH = (160, 6144)

HOST_CSV = Path("/data/aiter_configs/mi308x_bf16_gate_indexer.csv")
HOST_META = Path("/data/aiter_configs/mi308x_bf16_gate_indexer.meta")
HOST_SCRIPT = Path("/data/aiter_configs/gen_bf16_gate_indexer.py")
IMAGE_CSV = Path(
    "/sgl-workspace/aiter/aiter/configs/model_configs/mi308x_gfx942_bf16_tuned_gemm.csv"
)
TUNER = Path("/sgl-workspace/aiter/csrc/gemm_a16w16/gemm_a16w16_tune.py")

FIELDNAMES = [
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
    "libtype",
    "solidx",
    "splitK",
    "us",
    "kernelName",
    "err_ratio",
    "tflops",
    "bw",
]
PRIMARY = FIELDNAMES[:10]
REAL_LIBTYPES = {"opus", "hipblaslt", "asm"}

# Decode cuda-graph-bs-decode 1..32 plus draft=4 multiples (36=9*4, 40=10*4).
DECODE_M = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    13,
    16,
    20,
    24,
    28,
    32,
    36,
    40,
    48,
    52,
    64,
    80,
    96,
    128,
)
GRID_M = DECODE_M + (
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    16384,
)
# Cached-prefill leftovers from 2P2D Codex 12K logs (P0).
LEFTOVER_M = (
    210,
    243,
    258,
    280,
    420,
    467,
    473,
    483,
    540,
    560,
    738,
    748,
    905,
    986,
    1198,
    1609,
    1754,
    2817,
    2831,
    5900,
    7518,
)


def log(msg: str) -> None:
    print(f"[gate-indexer] {msg}", flush=True)


def key_of(row: dict) -> tuple:
    return tuple(str(row.get(c, "")) for c in PRIMARY)


def is_real(row: dict) -> bool:
    return (
        str(row.get("dtype", "")) == "torch.bfloat16"
        and str(row.get("libtype", "")) in REAL_LIBTYPES
    )


def as_int(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_meta(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def needed_m() -> list[int]:
    return sorted(set(GRID_M) | set(LEFTOVER_M))


def seed_rows(image_csv: Path) -> list[dict]:
    rows = [r for r in load_csv(image_csv) if is_real(r)]
    log(f"seed {image_csv} real_rows={len(rows)}")
    return rows


def clone_row(src: dict, m: int) -> dict:
    row = {k: src.get(k, "") for k in FIELDNAMES}
    row["M"] = str(m)
    row["gfx"] = GFX
    row["cu_num"] = str(src.get("cu_num") or CU)
    row["K"] = str(K)
    row["bias"] = src.get("bias") or "False"
    row["dtype"] = "torch.bfloat16"
    row["outdtype"] = src.get("outdtype") or "torch.bfloat16"
    row["scaleAB"] = src.get("scaleAB") or "False"
    row["bpreshuffle"] = src.get("bpreshuffle") or "False"
    return row


def nearest_seed(seeds: list[dict], m: int) -> dict:
    return min(seeds, key=lambda r: abs((as_int(r["M"]) or 0) - m))


def build_one(n: int, m: int, seeds: list[dict]) -> tuple[str, dict]:
    exact = [r for r in seeds if as_int(r["M"]) == m]
    if exact:
        return "exact", {k: exact[0].get(k, "") for k in FIELDNAMES}
    return "clone", clone_row(nearest_seed(seeds, m), m)


def generate_table(image_csv: Path, workers: int) -> tuple[list[dict], dict]:
    seed = seed_rows(image_csv)
    by_n: dict[int, list[dict]] = {}
    passthrough: list[dict] = []
    for row in seed:
        n = as_int(row.get("N", ""))
        k = as_int(row.get("K", ""))
        if k != K:
            passthrough.append(row)
            continue
        if n in N_GENERATE:
            by_n.setdefault(n, []).append(row)
        elif n in N_PASSTHROUGH:
            passthrough.append(row)

    tasks: list[tuple[int, int]] = []
    for n in N_GENERATE:
        if n not in by_n:
            log(f"WARNING no seed rows for N={n}; skip")
            continue
        # Never drop overlay M for this N (decode graph already tuned 3/5/6/7/9/10/...).
        seed_m = {as_int(r.get("M", "")) for r in by_n[n]}
        seed_m.discard(None)
        for m in sorted(set(needed_m()) | seed_m):
            tasks.append((n, m))

    counts = {"exact": 0, "clone": 0, "passthrough": len(passthrough)}
    built: list[dict] = []

    def _run(pair: tuple[int, int]) -> tuple[str, dict]:
        n, m = pair
        return build_one(n, m, by_n[n])

    workers = max(1, workers)
    log(f"generate tasks={len(tasks)} workers={workers} N={list(N_GENERATE)}")
    if workers == 1:
        results = [_run(t) for t in tasks]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gemm") as pool:
            futs = [pool.submit(_run, t) for t in tasks]
            for fut in as_completed(futs):
                results.append(fut.result())

    seen: set[tuple] = set()
    ordered: list[dict] = []
    for kind, row in results:
        counts[kind] += 1
        k = key_of(row)
        if k in seen:
            continue
        seen.add(k)
        ordered.append(row)
    for row in passthrough:
        k = key_of(row)
        if k in seen:
            continue
        seen.add(k)
        ordered.append(row)

    ordered.sort(
        key=lambda r: (
            as_int(r.get("N")) or 0,
            as_int(r.get("M")) or 0,
            str(r.get("libtype", "")),
        )
    )
    return ordered, counts


def persist_host(rows: list[dict], counts: dict, host_csv: Path, host_meta: Path, version: str) -> None:
    write_csv(host_csv, rows)
    n_by = {}
    for row in rows:
        n = str(row.get("N", ""))
        n_by[n] = n_by.get(n, 0) + 1
    meta = {
        "version": version,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": len(rows),
        "counts": counts,
        "rows_by_n": n_by,
        "csv": str(host_csv),
        "k": K,
        "n_generate": list(N_GENERATE),
        "host": os.uname().nodename,
    }
    write_meta(host_meta, meta)
    log(f"wrote {host_csv} rows={len(rows)} {counts} byN={n_by}")
    log(f"wrote {host_meta}")


def copy_self(script_src: Path | None) -> None:
    src = Path(script_src) if script_src else Path(__file__).resolve()
    if not src.is_file():
        return
    HOST_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != HOST_SCRIPT.resolve():
        shutil.copy2(src, HOST_SCRIPT)
        log(f"installed script {HOST_SCRIPT}")


def meta_current(host_meta: Path, version: str) -> bool:
    if not host_meta.is_file() or not HOST_CSV.is_file():
        return False
    try:
        payload = json.loads(host_meta.read_text())
    except json.JSONDecodeError:
        return False
    return str(payload.get("version", "")) == version and HOST_CSV.stat().st_size > 0


def cmd_generate(args: argparse.Namespace) -> int:
    copy_self(args.script_src)
    if meta_current(args.host_meta, args.version) and not args.force:
        log(f"host csv already version={args.version}, skip generate (use --force to rebuild)")
        return 0
    rows, counts = generate_table(args.image_csv, args.workers)
    if not rows:
        log("FATAL generated 0 rows")
        return 1
    persist_host(rows, counts, args.host_csv, args.host_meta, args.version)
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    if not args.host_csv.is_file():
        log(f"FATAL missing {args.host_csv}; run generate first")
        return 1
    args.image_csv.parent.mkdir(parents=True, exist_ok=True)
    bak = args.image_csv.with_suffix(args.image_csv.suffix + ".bak-image")
    if args.image_csv.is_file() and not bak.exists():
        shutil.copy2(args.image_csv, bak)
        log(f"backed up image overlay -> {bak}")
    shutil.copy2(args.host_csv, args.image_csv)
    log(f"OVERWRITE {args.image_csv} <- {args.host_csv} bytes={args.host_csv.stat().st_size}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    """Optional one-shot GPU tune. Not used by init. Uses aiter --mp."""
    if not TUNER.is_file():
        log(f"FATAL tuner missing {TUNER}")
        return 1
    if cmd_generate(args) != 0:
        return 1
    rows = load_csv(args.host_csv)
    need = [
        r
        for r in rows
        if as_int(r.get("N")) in N_GENERATE and as_int(r.get("K")) == K
    ]
    untune = Path("/tmp/mi308x_gate_indexer_untune.csv")
    tuned = Path("/tmp/mi308x_gate_indexer_tuned.csv")
    with untune.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=PRIMARY,
            extrasaction="ignore",
        )
        writer.writeheader()
        seen: set[tuple] = set()
        for row in need:
            k = key_of(row)
            if k in seen:
                continue
            seen.add(k)
            writer.writerow({c: row.get(c, "") for c in PRIMARY})
    libtypes = os.environ.get("AITER_BF16_GEMM_LIBTYPES", "hipblaslt,opus")
    cmd = [
        sys.executable,
        str(TUNER),
        "-i",
        str(untune),
        "-o",
        str(tuned),
        "--libtype",
        libtypes,
        "--mp",
        str(max(1, args.workers)),
        "--shape_grouped",
        "--warmup",
        "5",
        "--iters",
        "101",
        "--errRatio",
        "0.05",
    ]
    log(f"tune {' '.join(cmd)}")
    env = dict(os.environ)
    env.setdefault("SGLANG_USE_AITER", "1")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        log(f"FATAL tuner exit={proc.returncode}")
        return proc.returncode
    if not tuned.is_file():
        log(f"FATAL no tuner output {tuned}")
        return 1
    tuned_rows = [r for r in load_csv(tuned) if is_real(r)]
    by_key = {key_of(r): r for r in tuned_rows}
    merged: list[dict] = []
    seen = set()
    replaced = 0
    for row in rows:
        k = key_of(row)
        src = by_key.get(k, row)
        if k in by_key:
            replaced += 1
        if k in seen:
            continue
        seen.add(k)
        merged.append({c: src.get(c, "") for c in FIELDNAMES})
    persist_host(
        merged,
        {"tuned": replaced, "kept": len(merged) - replaced, "passthrough": 0},
        args.host_csv,
        args.host_meta,
        args.version,
    )
    log(f"tune replaced={replaced} total={len(merged)}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("generate", "install", "tune"))
    p.add_argument("--version", default=os.environ.get("AITER_BF16_GEMM_VERSION", VERSION))
    p.add_argument("--workers", type=int, default=int(os.environ.get("AITER_BF16_GEMM_WORKERS", "8")))
    p.add_argument("--host-csv", type=Path, default=HOST_CSV)
    p.add_argument("--host-meta", type=Path, default=None)
    p.add_argument("--image-csv", type=Path, default=IMAGE_CSV)
    p.add_argument("--script-src", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.host_meta is None:
        args.host_meta = args.host_csv.with_suffix(".meta")
    return args


def main() -> int:
    args = parse_args()
    log(f"cmd={args.command} version={args.version} workers={args.workers}")
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "install":
        return cmd_install(args)
    return cmd_tune(args)


if __name__ == "__main__":
    raise SystemExit(main())
