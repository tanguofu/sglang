#!/usr/bin/env python3
"""Build a corrected HTTPRoute patch that removes non-existent /liveness and /readiness paths.

Background:
- The worker (sglang server) does not have /liveness or /readiness endpoints.
- k8s uses /health (and /health_generate) for both liveness and readiness probes.
- /health and /health_generate already work via the router's catch-all rule.
- Including /liveness and /readiness in HTTPRoute caused them to route to the worker,
  which returns 401 (auth middleware rejects before 404 since path doesn't exist).
- This patch removes those two rules, leaving only the legitimate worker-direct paths.
"""

import json

# Paths that bypass the router and go directly to the worker service.
# These are endpoints the Rust router doesn't have in its hardcoded whitelist.
# All other paths (including /health, /v1/messages, /v1/chat/completions, /v1/responses)
# fall through to the catch-all router rule.
WORKER_DIRECT_PATHS = [
    "/v1/messages/count_tokens",
    "/metrics",
    "/get_server_info",
    "/get_model_info",
    "/flush_cache",
    "/engine_metrics",
]

WORKER_SERVICE = {
    "group": "",
    "kind": "Service",
    "name": "sglang-glm52-2tp8-sglang",
    "namespace": "kube-system",
    "port": 30000,
    "weight": 1,
}

ROUTER_SERVICE = {
    "group": "",
    "kind": "Service",
    "name": "sglang-glm52-2tp8-router",
    "namespace": "kube-system",
    "port": 30001,
    "weight": 1,
}


def build_rule(path_prefix: str, backend: dict) -> dict:
    return {
        "backendRefs": [backend],
        "matches": [{"path": {"type": "PathPrefix", "value": path_prefix}}],
    }


rules = [build_rule(p, WORKER_SERVICE) for p in WORKER_DIRECT_PATHS]
rules.append(build_rule("/", ROUTER_SERVICE))  # catch-all

patch = {"spec": {"rules": rules}}

with open("/tmp/httproute_patch_v2.json", "w") as f:
    json.dump(patch, f, indent=2)

print(f"Built patch with {len(rules)} rules ({len(WORKER_DIRECT_PATHS)} worker-direct + 1 catch-all)")
print(f"Removed: /liveness, /readiness (not real sglang endpoints)")
print(f"Worker-direct paths: {', '.join(WORKER_DIRECT_PATHS)}")
print()
print("Patch written to /tmp/httproute_patch_v2.json")
