#!/usr/bin/env python3
"""Patch the PoC-mode decode-1 STS: swap image to v0918-src-ckgemm + set CK env.

Idempotent. Strips resourceVersion/generation/managedFields for kubectl replace.
"""
import json
import subprocess
import sys

STS = "sglang-1p1d-decode-1"
NS = "kube-system"
IMAGE = "mirrors.tencent.com/ti-platform/sglang-glm52-308x:v0918-src-ckgemm"
ENV = {"name": "SGLANG_ROCM_USE_CK_GEMM_GFX942", "value": "1"}


def main():
    spec = json.loads(
        subprocess.run(
            ["kubectl", "get", "sts", STS, "-n", NS, "-o", "json"],
            capture_output=True, text=True, check=True,
        ).stdout
    )
    c = spec["spec"]["template"]["spec"]["containers"][0]
    c["image"] = IMAGE
    envs = c["env"]
    for e in envs:
        if e["name"] == ENV["name"]:
            e["value"] = ENV["value"]
            break
    else:
        envs.append(dict(ENV))
    for k in ("resourceVersion", "generation", "managedFields"):
        spec["metadata"].pop(k, None)
    out = "/tmp/decode-1-poc-ck.json"
    json.dump(spec, open(out, "w"))
    subprocess.run(["kubectl", "replace", "-f", out], check=True)
    print(f"[ok] {STS}: image={IMAGE} env {ENV['name']}={ENV['value']}")


if __name__ == "__main__":
    sys.exit(main())
