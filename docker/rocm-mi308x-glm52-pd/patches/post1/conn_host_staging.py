#!/usr/bin/env python3
"""Bake HIP host-staging into mooncake/conn.py (v0.5.17 image source).

Verified live path (kernel 5.4, no peermem):
  decode:  RDMA → host RAM → selective hipMemcpy H2D
  prefill: hipMemcpy D2H → RDMA from host
  both:    SGLANG_PD_HOST_STAGING=1

Does not overlay a whole conn.py (v0.5.17 send() has num_kv_tokens).
Runs the live-1p1d patch scripts in order. Idempotent if those scripts are.

Usage:
  python3 conn_host_staging.py /sgl-workspace/sglang/python/sglang/srt/disaggregation/mooncake/conn.py
"""
import subprocess
import sys
from pathlib import Path

CONN = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/sgl-workspace/sglang/python/sglang/srt/disaggregation/mooncake/conn.py"
)

HERE = Path(__file__).resolve().parent
CANDIDATES = [
    Path("/tmp/live-patches"),
    Path("/tmp/pd-host"),
    HERE,
    HERE.parent.parent / "live-1p1d" / "patches",
]

SCRIPTS = (
    "patch_host_staging.py",
    "patch_host_staging_v2.py",
    "patch_prefill_d2h.py",
)

root = next((p for p in CANDIDATES if (p / SCRIPTS[0]).exists()), None)
if root is None:
    raise SystemExit("conn_host_staging: patch_host_staging.py not found in " + str(CANDIDATES))

for name in SCRIPTS:
    path = root / name
    if not path.exists():
        raise SystemExit(f"conn_host_staging: missing {path}")
    print(f"conn_host_staging: running {path} on {CONN}")
    subprocess.check_call([sys.executable, str(path), str(CONN)])

src = CONN.read_text()
for marker in (
    "_copy_host_to_gpu",
    "refusing full-pool",
    "FIX(prefill-d2h-host-staging)",
):
    if marker not in src:
        raise SystemExit(f"conn_host_staging: missing marker {marker!r} in {CONN}")
print(f"[ok] {CONN}: host-staging + prefill D2H baked in")
