#!/usr/bin/env python3
"""Switch a 1P1D worker StatefulSet between production mode and PoC mode.

PoC mode runs a standalone (non-PD) sglang server on :31000 so the PD router
(health-checks :30000 only) keeps the worker out of rotation — zero user
traffic while the full PoC suite runs against the pod directly.

Usage:
  sts_poc_mode.py poc     <sts-name> [--image IMG]   # switch to PoC mode
  sts_poc_mode.py restore <sts-name>                 # restore backed-up spec
  sts_poc_mode.py show    <sts-name>                 # print current mode

The original spec is saved to /tmp/<sts-name>.prod.json on first switch.
Idempotent: switching an already-PoC STS is a no-op.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

POC_PORT = 31000
PROD_PORT = 30000
# PD-only launch args removed in PoC mode (substring match on the arg token).
PD_ARG_TOKENS = (
    "--disaggregation-transfer-backend",
    "--num-reserved-decode-tokens",
    "--load-balance-method",
    "--disaggregation-mode",
    "--disaggregation-ib-device",
    "--disaggregation-bootstrap-port",
)


def kubectl_json(sts: str) -> dict:
    out = subprocess.check_output(
        ["kubectl", "get", "sts", sts, "-n", "kube-system", "-o", "json"]
    )
    return json.loads(out)


def sglang_container(spec: dict) -> dict:
    for c in spec["spec"]["template"]["spec"]["containers"]:
        if c["name"] == "sglang":
            return c
    raise SystemExit("sglang container not found")


def is_poc(spec: dict) -> bool:
    c = sglang_container(spec)
    return f"--port {POC_PORT}" in c["args"][0]


def to_poc_args(launch: str) -> str:
    tokens = launch.split()
    kept = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if any(t == tok or t.startswith(tok + "=") for tok in PD_ARG_TOKENS):
            # value-taking arg without "=" consumes the next token too
            if "=" not in t and i + 1 < len(tokens):
                i += 2
            else:
                i += 1
            continue
        if t == "--port" and i + 1 < len(tokens) and tokens[i + 1] == str(PROD_PORT):
            kept += ["--port", str(POC_PORT)]
            i += 2
            continue
        kept.append(t)
        i += 1
    return " ".join(kept)


def apply_spec(spec: dict) -> None:
    subprocess.run(
        ["kubectl", "replace", "-f", "-"],
        input=json.dumps(spec).encode(),
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["poc", "restore", "show"])
    ap.add_argument("sts")
    ap.add_argument("--image", default=None, help="also swap the sglang container image")
    args = ap.parse_args()

    spec = kubectl_json(args.sts)
    backup = Path(f"/tmp/{args.sts}.prod.json")

    if args.mode == "show":
        print(f"{args.sts}: {'POC' if is_poc(spec) else 'PROD'} mode")
        return

    if args.mode == "restore":
        if not backup.exists():
            raise SystemExit(f"no backup at {backup}; restore manually from helm values")
        restored = json.loads(backup.read_text())
        apply_spec(restored)
        print(f"[ok] {args.sts}: restored production spec from {backup}")
        return

    # poc mode
    if is_poc(spec):
        print(f"[ok] {args.sts}: already in PoC mode")
        return
    if not backup.exists():
        backup.write_text(json.dumps(spec))
        print(f"[ok] saved production spec backup to {backup}")
    c = sglang_container(spec)
    script = c["args"][0]
    launch_idx = script.find("exec python3 -m sglang.launch_server")
    if launch_idx == -1:
        raise SystemExit("launch command not found in args")
    new_launch = to_poc_args(script[launch_idx:])
    c["args"][0] = script[:launch_idx] + new_launch
    for probe in ("livenessProbe", "readinessProbe"):
        if probe in c and c[probe].get("httpGet", {}).get("port") == PROD_PORT:
            c[probe]["httpGet"]["port"] = POC_PORT
    if args.image:
        c["image"] = args.image
        print(f"[ok] image -> {args.image}")
    apply_spec(spec)
    print(f"[ok] {args.sts}: PoC mode (standalone, :{POC_PORT}); router keeps it out of rotation")


if __name__ == "__main__":
    main()
