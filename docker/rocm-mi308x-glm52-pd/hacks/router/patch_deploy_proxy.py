#!/usr/bin/env python3
"""Patch sglang-1p1d-router deployment args to add tool_choice proxy."""
import json
import subprocess

NEW_ARGS = [
    "set -euo pipefail\n"
    'echo "=== SGLang PD Router (with tool_choice proxy) ==="\n'
    "cp /data/sglang_router_rs.abi3.so /sgl-workspace/sglang/sgl-model-gateway/bindings/python/src/sglang_router/sglang_router_rs.abi3.so\n"
    "# Start tool_choice normalizing proxy in background\n"
    "python3 /data/tc_proxy.py &\n"
    "PROXY_PID=$!\n"
    "sleep 1\n"
    'echo "Started tool_choice proxy (PID=$PROXY_PID) on :30011 -> :30012"\n'
    "# Start Rust router on port 30012 (proxy forwards 30011 -> 30012)\n"
    "exec python3 -m sglang_router.launch_router \\\n"
    "  --pd-disaggregation \\\n"
    "  --prefill http://21.151.225.144:30000 \\\n"
    "  --decode http://21.151.225.132:30000 \\\n"
    "  --host 0.0.0.0 --port 30012 \\\n"
    "  --prometheus-port 19096 \\\n"
    "  --model-path /data/model/glm52-fp8 \\\n"
    "  --health-check-timeout-secs 60 \\\n"
    "  --health-check-interval-secs 30 \\\n"
    "  --health-failure-threshold 10 \\\n"
    "  --health-success-threshold 2 \\\n"
    "  --cb-timeout-duration-secs 300 \\\n"
    "  --log-level info\n"
]

patch = [{"op": "replace", "path": "/spec/template/spec/containers/0/args", "value": NEW_ARGS}]

result = subprocess.run(
    [
        "kubectl", "patch", "deploy", "sglang-1p1d-router",
        "-n", "kube-system", "--type=json", "-p",
        json.dumps(patch),
    ],
    capture_output=True, text=True,
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("RC:", result.returncode)
