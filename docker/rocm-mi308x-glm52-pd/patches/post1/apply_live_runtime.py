#!/usr/bin/env python3
"""Bake live-1p1d runtime patches into the v0.5.17 image tree.

Runs after post1 semantic patches + conn_host_staging. Idempotent.
Host-staging Python remains env-gated (SGLANG_PD_HOST_STAGING=1 fallback).
GDR L2 flush is the default production path (HOST_STAGING!=1).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python/sglang")
CANDIDATES = [
    Path("/tmp/live-patches"),
    Path(__file__).resolve().parent.parent.parent / "live-1p1d" / "patches",
]
LIVE = next((p for p in CANDIDATES if (p / "patch_gdr_flush.py").exists()), None)
if LIVE is None:
    raise SystemExit("apply_live_runtime: patch_gdr_flush.py not found")

SCRIPTS = (
    "patch_gdr_flush.py",
    "patch_overlap_hip_wait.py",
    "patch_flydsl_mqa.py",
    "patch_decode_pd_health_flush.py",
    "patch_abort_noblock.py",
    "patch_bootstrap_room_scalar.py",
    "patch_responses_404.py",
    "patch_pd_send_timeout.py",
    "patch_scheduler_health.py",
)

for name in SCRIPTS:
    path = LIVE / name
    if not path.exists():
        raise SystemExit(f"apply_live_runtime: missing {path}")
    print(f"apply_live_runtime: {name}")
    subprocess.check_call([sys.executable, str(path)])

conn = (ROOT / "srt/disaggregation/mooncake/conn.py").read_text()
for marker in ("FIX(gdr-l2-flush)", "rdma_read_flush"):
    if marker not in conn:
        raise SystemExit(f"apply_live_runtime: missing {marker!r} in conn.py")
print("[ok] live runtime patches baked")
