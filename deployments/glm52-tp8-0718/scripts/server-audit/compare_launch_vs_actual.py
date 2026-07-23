#!/usr/bin/env python3
"""Compare launch args (from StatefulSet) vs actual server_info to find discrepancies."""
import json, subprocess, re

# Get launch args
cmd_args = [
    "kubectl", "get", "statefulset", "sglang-glm52-2tp8-sglang", "-n", "kube-system",
    "-o", "jsonpath={.spec.template.spec.containers[0].args}",
]
args_str = subprocess.check_output(cmd_args, text=True)

# Get server info
cmd_info = [
    "kubectl", "exec", "-n", "kube-system", "sglang-glm52-2tp8-sglang-0",
    "--", "/usr/bin/curl", "-sS", "--max-time", "15",
    "-H", "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}",
    "http://127.0.0.1:30000/get_server_info",
]
info = json.loads(subprocess.check_output(cmd_info, text=True))

# Parse --key value or --key=value from args
flags = {}
# args is a JSON array with one bash script string
script = json.loads(args_str)[0]
# Find all --flag "value" and --flag value patterns
for m in re.finditer(r'--([\w-]+)\s+"?([^"\s\\]+)"?', script):
    flags[m.group(1)] = m.group(2)

# Also handle --flag without value (boolean)
for m in re.finditer(r'--([\w-]+)(?=\s+--|\s*$|\s*\\)', script):
    if m.group(1) not in flags:
        flags[m.group(1)] = "true"

# Map launch flag names to server_info keys (most are 1:1 with - replaced by _)
def flag_to_key(flag):
    return flag.replace("-", "_")

print("=" * 75)
print("Launch args vs actual server_info (discrepancies highlighted)")
print("=" * 75)
print(f"\n{'flag':<35} {'launch value':<22} {'server_info value':<22} {'match?'}")
print("-" * 100)

discrepancies = []
for flag, launch_val in sorted(flags.items()):
    key = flag_to_key(flag)
    actual = info.get(key, "<missing>")
    # Normalize for comparison
    launch_str = str(launch_val).strip('"')
    actual_str = str(actual).strip('"')
    match = launch_str == actual_str
    marker = "OK" if match else "*** DIFF ***"
    if not match:
        discrepancies.append((flag, launch_str, actual_str))
    print(f"{flag:<35} {launch_str:<22} {actual_str:<22} {marker}")

print("\n" + "=" * 75)
if discrepancies:
    print(f"DISCREPANCIES ({len(discrepancies)}):")
    for flag, launch, actual in discrepancies:
        print(f"  --{flag}: launch='{launch}' vs actual='{actual}'")
else:
    print("No discrepancies")
