import os, json, base64, ssl, urllib.request

out = []

def read_file(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except:
        return None

def list_dir(path):
    try:
        return sorted(os.listdir(path))
    except:
        return []

# 1. Kernel modules
out.append("===== KERNEL MODULES (gpu/rdma) =====")
mods = read_file('/host/proc/modules')
if mods:
    found = [l for l in mods.splitlines() if any(k in l.lower() for k in ['amdgpu','amdkfd','ib_core','ib_cm','ib_ipoib','rdma','mlx'])]
    out.extend(found if found else ["(none found)"])
else:
    out.append("(cannot read /proc/modules)")

# 2. PCI devices
out.append("\n===== AMD GPU PCI devices (vendor 0x1002) =====")
out.append("\n===== Mellanox/RDMA NIC PCI devices (vendor 0x15b3) =====")
mlx_start = len(out) - 1
pci_dir = '/host/sys/bus/pci/devices/'
amd_count = 0
mlx_count = 0
if os.path.exists(pci_dir):
    for d in sorted(os.listdir(pci_dir)):
        dp = os.path.join(pci_dir, d)
        v = read_file(os.path.join(dp, 'vendor'))
        if v == '0x1002':
            dev = read_file(os.path.join(dp, 'device'))
            cls = read_file(os.path.join(dp, 'class'))
            out.insert(mlx_start, f"  {d} vendor={v} device={dev} class={cls}")
            amd_count += 1
        elif v == '0x15b3':
            dev = read_file(os.path.join(dp, 'device'))
            cls = read_file(os.path.join(dp, 'class'))
            out.append(f"  {d} vendor={v} device={dev} class={cls}")
            mlx_count += 1
if amd_count == 0:
    out.insert(mlx_start, "  (no AMD GPU devices found)")
if mlx_count == 0:
    out.append("  (no Mellanox devices found)")

# 3. InfiniBand/RDMA
out.append("\n===== /sys/class/infiniband (RDMA devices) =====")
ib = list_dir('/host/sys/class/infiniband/')
out.extend(ib if ib else ["(none / not found)"])

# 4. DRM/GPU
out.append("\n===== /sys/class/drm (GPU render nodes) =====")
drm = [x for x in list_dir('/host/sys/class/drm/') if 'render' in x or 'card' in x]
out.extend(drm if drm else ["(none)"])

# 5. /sys/module
out.append("\n===== /sys/module amdgpu/amdkfd/ib/rdma/mlx =====")
mods_sys = [x for x in list_dir('/host/sys/module/') if any(k in x.lower() for k in ['amdgpu','amdkfd','ib_','rdma','mlx'])]
out.extend(mods_sys if mods_sys else ["(none)"])

# 6. ROCm userspace
out.append("\n===== ROCm userspace =====")
rocm = list_dir('/host/opt/rocm/bin/')
out.extend(rocm[:5] if rocm else ["/opt/rocm not found"])

# 7. GPU device files
out.append("\n===== /dev/kfd and /dev/dri =====")
out.append(f"  /dev/kfd exists: {os.path.exists('/host/dev/kfd')}")
dri = list_dir('/host/dev/dri/')
out.append(f"  /dev/dri/: {dri if dri else '(not found)'}")

# 8. InfiniBand device files
out.append("\n===== /dev/infiniband =====")
ibdev = list_dir('/host/dev/infiniband/')
out.append(f"  /dev/infiniband/: {ibdev if ibdev else '(not found)'}")

# 9. amdgpu firmware
out.append("\n===== amdgpu firmware =====")
fw = list_dir('/host/lib/firmware/amdgpu/')
out.extend(fw[:3] if fw else ["/lib/firmware/amdgpu not found"])

out.append("\n===== DONE =====")

result = "\n".join(out)
result_b64 = base64.b64encode(result.encode()).decode()

# Create ConfigMap via API
token = read_file('/var/run/secrets/kubernetes.io/serviceaccount/token')
ns = read_file('/var/run/secrets/kubernetes.io/serviceaccount/namespace')
api_host = os.environ.get('KUBERNETES_SERVICE_HOST', 'kubernetes.default.svc')
api_port = os.environ.get('KUBERNETES_SERVICE_PORT', '443')

cm_name = os.environ.get('CM_NAME', 'drv-check')
cm = {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {"name": cm_name},
    "binaryData": {"output": result_b64}
}
data = json.dumps(cm).encode()
req = urllib.request.Request(
    f"https://{api_host}:{api_port}/api/v1/namespaces/{ns}/configmaps",
    data=data,
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    method="POST"
)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
try:
    resp = urllib.request.urlopen(req, context=ctx, timeout=15)
    print(f"ConfigMap created: HTTP {resp.status}")
except Exception as e:
    print(f"ConfigMap creation FAILED: {e}")
print(result)
