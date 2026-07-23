import json
import subprocess

# Get current HTTPRoute
result = subprocess.run(
    ['kubectl', 'get', 'httproute', 'sglang-glm52-2tp8-sglang', '-n', 'kube-system', '-o', 'json'],
    capture_output=True, text=True, check=True
)
route = json.loads(result.stdout)

# Build the new rules: specific paths first, then the catch-all
# Worker-direct paths (bypass router for endpoints the Rust router doesn't handle)
worker_direct_paths = [
    '/v1/messages/count_tokens',
    '/metrics',
    '/get_server_info',
    '/get_model_info',
    '/flush_cache',
    '/engine_metrics',
    '/liveness',
    '/readiness',
]

# Add a rule for each worker-direct path
new_rules = []
for path in worker_direct_paths:
    new_rules.append({
        'matches': [{
            'path': {
                'type': 'PathPrefix',
                'value': path
            }
        }],
        'backendRefs': [{
            'group': '',
            'kind': 'Service',
            'name': 'sglang-glm52-2tp8-sglang',
            'namespace': 'kube-system',
            'port': 30000,
            'weight': 1
        }]
    })

# Append the existing catch-all rule (router)
new_rules.extend(route['spec']['rules'])

patch = {
    'spec': {
        'rules': new_rules
    }
}

with open('/tmp/httproute_patch.json', 'w') as f:
    json.dump(patch, f, indent=2)

print(f'Patch written with {len(new_rules)} rules ({len(worker_direct_paths)} worker-direct + 1 catch-all)')
print('Worker-direct paths:')
for p in worker_direct_paths:
    print(f'  {p} -> sglang-glm52-2tp8-sglang:30000 (w1 worker)')
print('Catch-all: / -> sglang-glm52-2tp8-router:30001 (router)')
