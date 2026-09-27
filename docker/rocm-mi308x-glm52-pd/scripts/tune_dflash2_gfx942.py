#!/usr/bin/env python3
"""Prepare and run AIter GEMM tuning for DFlash2 GLM-5.3 prefill shapes."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BF16_LOG_PATTERN = re.compile(
    r"shape is M:(?P<m>\d+), N:(?P<n>\d+), K:(?P<k>\d+) "
    r"dtype='(?P<dtype>[^']+)' otype='(?P<otype>[^']+)' "
    r"bias=(?P<bias>True|False), scaleAB=(?P<scale_ab>True|False), "
    r"bpreshuffle=(?P<bpreshuffle>True|False), not found tuned config"
)
A8W8_LOG_PATTERN = re.compile(
    r"shape is M:(?P<m>\d+), N:(?P<n>\d+), K:(?P<k>\d+), "
    r"not found tuned config"
)
BF16_TUNER = Path(
    "/sgl-workspace/aiter/csrc/gemm_a16w16/gemm_a16w16_tune.py"
)
A8W8_TUNER = Path(
    "/sgl-workspace/aiter/csrc/ck_gemm_a8w8_blockscale/"
    "gemm_a8w8_blockscale_tune.py"
)


@dataclass(frozen=True)
class ObservedGemm:
    gemm_type: str
    m: int
    n: int
    k: int
    dtype: str | None = None
    otype: str | None = None
    bias: bool = False
    scale_ab: bool = False
    bpreshuffle: bool = False


@dataclass(frozen=True)
class ShapeGroup:
    gemm_type: str
    n: int
    k: int
    raw_m: tuple[int, ...]
    dtype: str | None
    otype: str | None
    bias: bool
    scale_ab: bool
    bpreshuffle: bool


@dataclass(frozen=True)
class TuningRequest:
    gemm_type: str
    m: int
    n: int
    k: int
    dtype: str | None
    otype: str | None
    bias: bool
    scale_ab: bool
    bpreshuffle: bool


@dataclass(frozen=True)
class PreparedTuningFiles:
    bf16_input: Path
    a8w8_input: Path
    summary_path: Path
    summary: dict[str, Any]


def _parse_bool(value: str) -> bool:
    return value == "True"


def parse_missing_configs(log_text: str) -> list[ObservedGemm]:
    records: list[ObservedGemm] = []
    for line in log_text.splitlines():
        match = BF16_LOG_PATTERN.search(line)
        if match is not None:
            records.append(
                ObservedGemm(
                    gemm_type="bf16",
                    m=int(match["m"]),
                    n=int(match["n"]),
                    k=int(match["k"]),
                    dtype=match["dtype"],
                    otype=match["otype"],
                    bias=_parse_bool(match["bias"]),
                    scale_ab=_parse_bool(match["scale_ab"]),
                    bpreshuffle=_parse_bool(match["bpreshuffle"]),
                )
            )
            continue

        match = A8W8_LOG_PATTERN.search(line)
        if match is not None:
            records.append(
                ObservedGemm(
                    gemm_type="a8w8",
                    m=int(match["m"]),
                    n=int(match["n"]),
                    k=int(match["k"]),
                )
            )
    return records


def group_missing_configs(
    records: Sequence[ObservedGemm],
) -> list[ShapeGroup]:
    grouped_raw_m: dict[tuple[Any, ...], set[int]] = {}
    for record in records:
        key = (
            record.gemm_type,
            record.n,
            record.k,
            record.dtype,
            record.otype,
            record.bias,
            record.scale_ab,
            record.bpreshuffle,
        )
        grouped_raw_m.setdefault(key, set()).add(record.m)

    groups = [
        ShapeGroup(
            gemm_type=key[0],
            n=key[1],
            k=key[2],
            raw_m=tuple(sorted(raw_m)),
            dtype=key[3],
            otype=key[4],
            bias=key[5],
            scale_ab=key[6],
            bpreshuffle=key[7],
        )
        for key, raw_m in grouped_raw_m.items()
    ]
    return sorted(groups, key=lambda group: (group.gemm_type, group.n, group.k))


def build_tuning_requests(
    groups: Sequence[ShapeGroup],
    pad_m: Callable[[int, int, int, int], int],
    include_raw_m: bool = False,
    include_gl1: bool = False,
) -> list[TuningRequest]:
    requests: list[TuningRequest] = []
    for group in groups:
        padded_m: set[int] = set()
        if include_raw_m:
            padded_m.update(group.raw_m)
        graph_launches = (0, 1) if include_gl1 else (0,)
        for raw_m in group.raw_m:
            padded_m.update(
                int(pad_m(raw_m, group.n, group.k, graph_launch))
                for graph_launch in graph_launches
            )

        requests.extend(
            TuningRequest(
                gemm_type=group.gemm_type,
                m=m,
                n=group.n,
                k=group.k,
                dtype=group.dtype,
                otype=group.otype,
                bias=group.bias,
                scale_ab=group.scale_ab,
                bpreshuffle=group.bpreshuffle,
            )
            for m in sorted(padded_m)
        )
    return sorted(
        requests, key=lambda request: (request.gemm_type, request.m, request.n, request.k)
    )


def _write_csv(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue()


def render_untuned_csvs(
    requests: Sequence[TuningRequest], gfx: str, cu_num: int
) -> tuple[str, str]:
    bf16_rows = [
        (
            gfx,
            cu_num,
            request.m,
            request.n,
            request.k,
            request.bias,
            request.dtype,
            request.otype,
            request.scale_ab,
            request.bpreshuffle,
        )
        for request in requests
        if request.gemm_type == "bf16"
    ]
    a8w8_rows = [
        (request.m, request.n, request.k)
        for request in requests
        if request.gemm_type == "a8w8"
    ]

    bf16_csv = _write_csv(
        (
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
        ),
        bf16_rows,
    )
    a8w8_csv = _write_csv(("M", "N", "K"), a8w8_rows)
    return bf16_csv, a8w8_csv


def filter_tuned_rows(
    tuned_csv: str,
    gemm_type: str,
    gfx: str,
    cu_num: int,
    max_err_ratio: float,
) -> str:
    if gemm_type not in {"bf16", "a8w8"}:
        raise ValueError(f"unsupported gemm type: {gemm_type}")

    error_column = "err_ratio" if gemm_type == "bf16" else "errRatio"
    reader = csv.DictReader(io.StringIO(tuned_csv))
    if reader.fieldnames is None:
        raise ValueError("tuned CSV has no header")
    for required_column in ("gfx", "cu_num", "us", "kernelName", error_column):
        if required_column not in reader.fieldnames:
            raise ValueError(f"tuned CSV is missing column: {required_column}")

    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=reader.fieldnames, lineterminator="\n"
    )
    writer.writeheader()
    for row in reader:
        if row["gfx"] != gfx or row["cu_num"] != str(cu_num):
            continue
        kernel_name = row["kernelName"]
        if not kernel_name or kernel_name.lower() == "none":
            continue
        try:
            elapsed_us = float(row["us"])
            error_ratio = float(row[error_column])
        except (TypeError, ValueError):
            continue
        if elapsed_us <= 0 or error_ratio > max_err_ratio:
            continue
        writer.writerow(row)
    return output.getvalue()


def load_aiter_pad_m() -> Callable[[int, int, int, int], int]:
    from aiter.ops.gemm_op_common import get_padded_m

    return get_padded_m


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def aiter_padded_m(m: int, n: int, k: int, graph_launch: int) -> int:
    if graph_launch == 0:
        if m <= 256:
            return (m + 15) // 16 * 16
        if m <= 1024:
            return (m + 31) // 32 * 32
        if m <= 4096:
            return (m + 63) // 64 * 64
        return (m + 127) // 128 * 128
    if graph_launch == 1:
        if m > 8192 and n > 4096:
            return 8192
        return _next_power_of_two(m)
    raise ValueError(f"unsupported graph launch: {graph_launch}")


def prepare_files(
    log_path: Path,
    output_dir: Path,
    gfx: str,
    cu_num: int,
    pad_m: Callable[[int, int, int, int], int],
    include_raw_m: bool = False,
    include_gl1: bool = False,
) -> PreparedTuningFiles:
    records = parse_missing_configs(log_path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"no missing tuned-config lines found in {log_path}")

    groups = group_missing_configs(records)
    requests = build_tuning_requests(
        groups,
        pad_m=pad_m,
        include_raw_m=include_raw_m,
        include_gl1=include_gl1,
    )
    bf16_csv, a8w8_csv = render_untuned_csvs(requests, gfx=gfx, cu_num=cu_num)

    output_dir.mkdir(parents=True, exist_ok=True)
    bf16_input = output_dir / "bf16_untuned.csv"
    a8w8_input = output_dir / "a8w8_untuned.csv"
    summary_path = output_dir / "summary.json"
    bf16_input.write_text(bf16_csv, encoding="utf-8")
    a8w8_input.write_text(a8w8_csv, encoding="utf-8")

    summary = {
        "missing_lines": len(records),
        "bf16_shapes": sum(request.gemm_type == "bf16" for request in requests),
        "a8w8_shapes": sum(request.gemm_type == "a8w8" for request in requests),
        "groups": [
            {
                "gemm_type": group.gemm_type,
                "n": group.n,
                "k": group.k,
                "raw_m_min": min(group.raw_m),
                "raw_m_max": max(group.raw_m),
                "raw_m_count": len(group.raw_m),
            }
            for group in groups
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return PreparedTuningFiles(
        bf16_input=bf16_input,
        a8w8_input=a8w8_input,
        summary_path=summary_path,
        summary=summary,
    )


def build_tuner_command(
    gemm_type: str,
    python_executable: str,
    tuner_path: Path,
    input_path: Path,
    output_path: Path,
    mp: int,
    warmup: int,
    iters: int,
    err_ratio: float,
    timeout: int,
    bf16_libtypes: str,
    a8w8_libtype: str,
) -> list[str]:
    command = [
        python_executable,
        str(tuner_path),
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "--mp",
        str(mp),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--errRatio",
        str(err_ratio),
        "--timeout",
        str(timeout),
        "--shape_grouped",
        "--splitK",
    ]
    if gemm_type == "bf16":
        command.extend(["--libtype", bf16_libtypes])
    elif gemm_type == "a8w8":
        command.extend(["--libtype", a8w8_libtype])
    else:
        raise ValueError(f"unsupported gemm type: {gemm_type}")
    return command


def _has_data_rows(path: Path) -> bool:
    return len(path.read_text(encoding="utf-8").splitlines()) > 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path)
    parser.add_argument(
        "--mode", choices=("prepare", "tune"), default="prepare"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/dflash2-aiter-tuning"))
    parser.add_argument("--gfx", default="gfx942")
    parser.add_argument("--cu-num", type=int, default=80)
    parser.add_argument("--max-err-ratio", type=float, default=0.05)
    parser.add_argument("--include-raw-m", action="store_true")
    parser.add_argument(
        "--include-gl1",
        action="store_true",
        help="Also tune power-of-two fallback buckets; gl0 coverage is sufficient "
        "for observed shapes",
    )
    parser.add_argument(
        "--pad-m-source",
        choices=("local", "aiter"),
        default="local",
        help="Use the source-verified local bucket rule or the installed AIter operator",
    )
    parser.add_argument("--mp", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=101)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--bf16-libtypes", default="torch,skinny,triton,opus"
    )
    parser.add_argument("--a8w8-libtype", default="both", choices=("ck", "cktile", "both", "all"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    pad_m = (
        load_aiter_pad_m()
        if args.pad_m_source == "aiter"
        else aiter_padded_m
    )
    prepared = prepare_files(
        log_path=args.log_file,
        output_dir=args.output_dir,
        gfx=args.gfx,
        cu_num=args.cu_num,
        pad_m=pad_m,
        include_raw_m=args.include_raw_m,
        include_gl1=args.include_gl1,
    )
    print(json.dumps(prepared.summary, indent=2))
    if args.mode == "prepare":
        return 0

    for gemm_type, input_path, tuner_path in (
        ("bf16", prepared.bf16_input, BF16_TUNER),
        ("a8w8", prepared.a8w8_input, A8W8_TUNER),
    ):
        if not _has_data_rows(input_path):
            continue
        tuned_path = args.output_dir / f"{gemm_type}_tuned.csv"
        command = build_tuner_command(
            gemm_type=gemm_type,
            python_executable=sys.executable,
            tuner_path=tuner_path,
            input_path=input_path,
            output_path=tuned_path,
            mp=args.mp,
            warmup=args.warmup,
            iters=args.iters,
            err_ratio=args.max_err_ratio,
            timeout=args.timeout,
            bf16_libtypes=args.bf16_libtypes,
            a8w8_libtype=args.a8w8_libtype,
        )
        subprocess.run(command, check=True)
        filtered_rows = filter_tuned_rows(
            tuned_path.read_text(encoding="utf-8"),
            gemm_type=gemm_type,
            gfx=args.gfx,
            cu_num=args.cu_num,
            max_err_ratio=args.max_err_ratio,
        )
        filtered_path = args.output_dir / f"{gemm_type}_{args.gfx}_rows.csv"
        filtered_path.write_text(filtered_rows, encoding="utf-8")
        print(f"wrote filtered rows to {filtered_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
