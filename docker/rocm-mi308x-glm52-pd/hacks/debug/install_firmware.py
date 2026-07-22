import os, json, base64, ssl, urllib.request, subprocess, shutil

out = []

def run(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        s = r.stdout + (("\n" + r.stderr) if r.stderr.strip() else "")
        return s.strip()
    except subprocess.TimeoutExpired:
        return f"(TIMEOUT after {timeout}s)"
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

def download(url, dest):
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            with open(dest, "wb") as f:
                downloaded = 0
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
            size = os.path.getsize(dest)
            return f"OK ({size} bytes)"
    except Exception as e:
        return f"FAILED: {e}"

kernel = read_file("/host/proc/sys/kernel/osrelease") or "unknown"
FW_URL = "https://repo.radeon.com/amdgpu/30.30.4/el/9.7/main/x86_64/amdgpu-dkms-firmware-30.30.4.0.30300400-2341068.el9.noarch.rpm"
FW_RPM = "amdgpu-dkms-firmware-30.30.4.0.30300400-2341068.el9.noarch.rpm"

out.append("===== HOST DISK SPACE =====")
df = run("chroot /host df -h / /tmp 2>/dev/null")
out.append(df)

out.append("\n===== CHECK EXISTING FIRMWARE =====")
rpm_q = run("chroot /host rpm -q amdgpu-dkms-firmware 2>/dev/null")
out.append(f"  rpm -q: {rpm_q}")
fw_exists = os.path.exists("/host/lib/firmware/amdgpu")
out.append(f"  /lib/firmware/amdgpu exists: {fw_exists}")
if fw_exists:
    fw_files = list_dir("/host/lib/firmware/amdgpu")
    out.append(f"  /lib/firmware/amdgpu file count: {len(fw_files)}")
    if fw_files:
        out.append(f"  sample files: {fw_files[:5]}")

out.append("\n===== DOWNLOAD FIRMWARE RPM =====")
os.makedirs("/host/tmp", exist_ok=True)
dest = f"/host/tmp/{FW_RPM}"
out.append(f"  URL: {FW_URL}")
out.append(f"  dest: {dest}")
dl_result = download(FW_URL, dest)
out.append(f"  download: {dl_result}")

if "OK" in dl_result:
    out.append("\n===== INSTALL FIRMWARE RPM =====")
    # Remove old package first if installed but broken, then reinstall
    install = run(f"chroot /host rpm -Uvh --force --nodeps /tmp/{FW_RPM} 2>&1", timeout=180)
    out.append(install if install else "(no output)")

    out.append("\n===== VERIFY FIRMWARE =====")
    fw_exists2 = os.path.exists("/host/lib/firmware/amdgpu")
    out.append(f"  /lib/firmware/amdgpu exists: {fw_exists2}")
    if fw_exists2:
        fw_files2 = list_dir("/host/lib/firmware/amdgpu")
        out.append(f"  file count: {len(fw_files2)}")
        out.append(f"  sample: {fw_files2[:5]}")
    rpm_q2 = run("chroot /host rpm -q amdgpu-dkms-firmware 2>/dev/null")
    out.append(f"  rpm -q: {rpm_q2}")
else:
    out.append("\n===== SKIPPED INSTALL (download failed) =====")

# Cleanup
try:
    os.remove(dest)
except:
    pass

out.append("\n===== DONE =====")

result = "\n".join(out)
result_b64 = base64.b64encode(result.encode()).decode()

token = read_file("/var/run/secrets/kubernetes.io/serviceaccount/token")
ns = read_file("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
api_host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
api_port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
cm_name = os.environ.get("CM_NAME", "fw-install")

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
