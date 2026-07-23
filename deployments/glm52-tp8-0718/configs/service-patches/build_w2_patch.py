import json

# Build w2 patch (same params as w1)
with open('/tmp/w2_args_original.sh', 'r') as f:
    content = f.read()

# Apply the same 4 substitutions
modified = content
modified = modified.replace('--mem-fraction-static 0.88', '--mem-fraction-static 0.82')
modified = modified.replace('--chunked-prefill-size 32768', '--chunked-prefill-size 131072')
modified = modified.replace('--prefill-max-requests 32 --max-prefill-tokens 32768', '--prefill-max-requests 32 --max-prefill-tokens 131072')
modified = modified.replace('--watchdog-timeout 3600', '--watchdog-timeout 1200')

# Verify changes
assert '--mem-fraction-static 0.82' in modified, "mem-fraction not patched"
assert '--chunked-prefill-size 131072' in modified, "chunked-prefill not patched"
assert '--max-prefill-tokens 131072' in modified, "max-prefill-tokens not patched"
assert '--watchdog-timeout 1200' in modified, "watchdog not patched"
assert '--mem-fraction-static 0.88' not in modified, "old mem-fraction still present"
assert '--chunked-prefill-size 32768' not in modified, "old chunked-prefill still present"

patch = {
    "spec": {
        "template": {
            "spec": {
                "containers": [
                    {
                        "name": "sglang",
                        "args": [modified]
                    }
                ]
            }
        }
    }
}

with open('/tmp/w2_patch.json', 'w') as f:
    json.dump(patch, f)

print(f"w2 patch written ({len(json.dumps(patch))} bytes)")
print("=== Verified params in modified args ===")
for line in modified.splitlines():
    if any(p in line for p in ['mem-fraction-static', 'chunked-prefill-size', 'max-prefill-tokens', 'watchdog-timeout']):
        print(f"  {line.strip()}")
