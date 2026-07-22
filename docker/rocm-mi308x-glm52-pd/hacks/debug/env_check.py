import os, json, base64, ssl, urllib.request, subprocess, socket

out = []

def run(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout + (r.stderr if r.stderr else "")
    except Exception as e:
        return f"(error: {e})"

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

kernel = read_file('/host/proc/sys/kernel/osrelease') or 'unknown'
out.append(f"===== OS & KERNEL =====")
out.append(f"  kernel: {kernel}")
out.append(f"  os-release:")
osrel = read_file('/host/etc/os-release')
if osrel:
    for line in osrel.splitlines():
        out.append(f"    {line}")

out.append(f"\n===== EXISTING amdgpu/rocm/ofa PACKAGES =====")
pkgs = run(f"chroot /host rpm -qa 2>/dev/null | grep -iE 'amdgpu|rocm|amd-smi|ofa|mlnx|dkms' | sort")
out.append(pkgs if pkgs.strip() else "(none found)")

out.append(f"\n===== KERNEL HEADERS / DEVEL =====")
kheaders = list_dir(f'/host/usr/src/kernels/')
out.append(f"  /usr/src/kernels/: {kheaders if kheaders else '(not found)'}")
build_link = os.path.exists(f'/host/lib/modules/{kernel}/build')
out.append(f"  /lib/modules/{kernel}/build exists: {build_link}")
if build_link:
    build_target = run(f"readlink -f /host/lib/modules/{kernel}/build 2>/dev/null").strip()
    out.append(f"    -> {build_target}")

out.append(f"\n===== ofa_kernel (RDMA peer memory) =====")
ofa = list_dir('/host/usr/src/ofa_kernel/')
out.append(f"  /usr/src/ofa_kernel/: {ofa if ofa else '(not found)'}")
if ofa:
    for arch in ofa:
        kdir = f'/host/usr/src/ofa_kernel/{arch}/{kernel}'
        if os.path.exists(kdir):
            symvers = os.path.exists(f'{kdir}/Module.symvers')
            out.append(f"    {arch}/{kernel}/Module.symvers exists: {symvers}")

out.append(f"\n===== BUILD TOOLS =====")
for tool in ['gcc', 'g++', 'make', 'cc', 'dkms', 'rpm', 'dnf', 'yum']:
    p = run(f"chroot /host which {tool} 2>/dev/null").strip()
    out.append(f"  {tool}: {p if p else '(not found)'}")
gccver = run("chroot /host gcc --version 2>/dev/null | head -1").strip()
out.append(f"  gcc version: {gccver if gccver else '(n/a)'}")

out.append(f"\n===== DISK SPACE =====")
for line in read_file('/host/proc/mounts').splitlines():
    parts = line.split()
    if len(parts) >= 2 and parts[1] in ['/', '/data', '/data1', '/var']:
        out.append(f"  {parts[1]} -> {parts[0]} ({parts[2]})")

out.append(f"\n===== NETWORK: repo.radeon.com =====")
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect(("repo.radeon.com", 443))
    out.append("  repo.radeon.com:443 reachable: YES")
    s.close()
except Exception as e:
    out.append(f"  repo.radeon.com:443 reachable: NO ({e})")

out.append(f"\n===== KERNEL MODULES DIR =====")
moddir = list_dir(f'/host/lib/modules/{kernel}')
out.append(f"  /lib/modules/{kernel}/: {len(moddir)} entries")
extra = list_dir(f'/host/lib/modules/{kernel}/extra')
out.append(f"  /lib/modules/{kernel}/extra/: {extra if extra else '(empty/not found)'}")

out.append(f"\n===== CURRENT amdgpu MODULE STATUS =====")
mods = read_file('/host/proc/modules')
if mods:
    amd = [l for l in mods.splitlines() if 'amdgpu' in l or 'amdkfd' in l]
    out.extend(amd if amd else ["  amdgpu/amdkfd NOT loaded"])

out.append(f"\n===== FIRMWARE =====")
fw = list_dir('/host/lib/firmware/amdgpu/')
out.append(f"  /lib/firmware/amdgpu/: {len(fw)} files" if fw else "  /lib/firmware/amdgpu/: (not found)")

out.append(f"\n===== ROCm =====")
rocm = list_dir('/host/opt/rocm/')
out.append(f"  /opt/rocm/: {rocm[:5] if rocm else '(not found)'}")

out.append(f"\n===== YUM REPOS =====")
repos = list_dir('/host/etc/yum.repos.d/')
out.append(f"  /etc/yum.repos.d/: {repos if repos else '(not found)'}")

out.append(f"\n===== DKMS STATUS =====")
dkms = run("chroot /host dkms status 2>/dev/null")
out.append(dkms if dkms.strip() else "(dkms not available or no modules)")

out.append(f"\n===== DONE =====")

result = "\n".join(out)
result_b64 = base64.b64encode(result.encode()).decode()

token = read_file('/var/run/secrets/kubernetes.io/serviceaccount/token')
ns = read_file('/var/run/secrets/kubernetes.io/serviceaccount/namespace')
api_host = os.environ.get('KUBERNETES_SERVICE_HOST', 'kubernetes.default.svc')
api_port = os.environ.get('KUBERNETES_SERVICE_PORT', '443')
cm_name = os.environ.get('CM_NAME', 'env-check')

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
