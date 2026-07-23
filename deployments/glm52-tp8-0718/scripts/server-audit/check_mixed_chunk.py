#!/usr/bin/env python3
import json, subprocess, re

args_str = subprocess.check_output([
    "kubectl", "get", "statefulset", "sglang-glm52-2tp8-sglang", "-n", "kube-system",
    "-o", "jsonpath={.spec.template.spec.containers[0].args}",
], text=True)
script = json.loads(args_str)[0]

# Find enable-mixed-chunk context (capture line)
for flag in ["enable-mixed-chunk", "disable-mixed-chunk"]:
    m = re.search(r'(--' + flag + r'[^\n\\]*)', script)
    if m:
        print(f"Found --{flag}:", repr(m.group(1)))
    else:
        print(f"NOT FOUND: --{flag}")

# Check server logs for mixed-chunk messages
print("\n--- Server log messages mentioning mixed_chunk ---")
log = subprocess.run([
    "kubectl", "logs", "-n", "kube-system", "sglang-glm52-2tp8-sglang-0", "--tail=5000"
], capture_output=True, text=True)
for line in log.stdout.splitlines():
    if "mixed" in line.lower() and "chunk" in line.lower():
        print(line[:200])
