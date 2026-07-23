import json, subprocess, sys

NS = "kube-system"
ROUTER = "pd-router-172"
NEW_IMAGE = "mirrors.tencent.com/ti-platform/sglang-glm52-308x-pd-router:responses-0714"

# Wheel is baked into the image — no pip install needed at runtime.
CMD = (
    "set -euo pipefail\n"
    'echo "=== SGLang PD Router (responses) — wheel baked in ==="\n'
    "exec python3 -m sglang_router.launch_router \\\n"
    "  --pd-disaggregation \\\n"
    "  --prefill http://21.151.225.144:30000 \\\n"
    "  --decode http://21.151.225.132:30000 \\\n"
    "  --host 0.0.0.0 --port 30001 \\\n"
    "  --model-path /data/model/glm52-fp8 \\\n"
    "  --log-level info\n"
)

obj = json.loads(subprocess.check_output(["kubectl", "get", "pod", ROUTER, "-n", NS, "-o", "json"]))
c = obj["spec"]["containers"][0]
c["image"] = NEW_IMAGE
c["imagePullPolicy"] = "Always"
c["command"] = ["/bin/bash", "-c"]
c["args"] = [CMD]
print(f"{ROUTER}: image={NEW_IMAGE} (wheel baked in, no runtime patch)")
for k in ("uid", "resourceVersion", "creationTimestamp", "managedFields", "generation", "selfLink"):
    obj["metadata"].pop(k, None)
obj.pop("status", None)
r = subprocess.run(["kubectl", "replace", "--force", "-f", "-"],
                   input=json.dumps(obj), capture_output=True, text=True)
print(r.stdout.strip())
if r.returncode != 0:
    print("ERR:", r.stderr.strip()[:300]); sys.exit(1)
