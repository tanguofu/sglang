#!/usr/bin/env python3
"""
add-init-container.py — Add initContainer with env checks + GEMM CSV persistence
to sglang-1p1d-prefill and sglang-1p1d-decode StatefulSets.

Usage:
    python3 add-init-container.py           # patch both StatefulSets
    python3 add-init-container.py --dry-run  # show changes without applying
    python3 add-init-container.py --prefill  # only patch prefill
    python3 add-init-container.py --decode   # only patch decode
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

NAMESPACE = "kube-system"
STS_PREFIX = "sglang-1p1d"
INIT_CONTAINER_NAME = "env-check-aiter-init"
SHARED_VOLUME_NAME = "shared-init"
SHARED_VOLUME_MOUNT = "/shared"

# The initContainer script (kept inline so the patch is self-contained).
# This is a copy of scripts/init-env-check.sh — update both if changing.
INIT_SCRIPT = r"""set -euo pipefail

echo "========== Environment Pre-Flight Check =========="
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Hostname:  $(hostname)"
echo ""

# 1. GPU Detection
echo "--- GPU Detection ---"
GPU_COUNT=$(rocminfo 2>/dev/null | grep -c "^  Name:.*gfx" || true)
echo "GPU count: ${GPU_COUNT}"
if [ "${GPU_COUNT}" -lt 1 ]; then
  echo "FATAL: No GPU detected via rocminfo" >&2
  exit 1
fi
rocminfo 2>/dev/null | grep -E "Marketing Name:|Name:.*gfx|Compute Unit:" | head -12
echo ""

# 2. Driver Version
echo "--- Driver Version ---"
rocm-smi --showdriverversion 2>&1 | grep -i "driver version" || echo "(unable to query driver version)"
echo ""

# 3. Firmware Version (GPU 0 only)
echo "--- Firmware Version (GPU 0) ---"
rocm-smi --showfwinfo 2>&1 | grep "GPU\[0\]" | head -10 || echo "(unable to query firmware info)"
echo ""

# 4. VRAM Occupancy
echo "--- VRAM Occupancy ---"
rocm-smi --showmeminfo vram 2>&1 | grep -E "GPU\[|VRAM Total" | head -20
VRAM_USED=$(rocm-smi --showmeminfo vram 2>&1 | grep "GPU\[0\].*Used" | awk '{print $NF}' || true)
VRAM_TOTAL=$(rocm-smi --showmeminfo vram 2>&1 | grep "GPU\[0\].*Total Memory" | awk '{print $NF}' || true)
if [ -n "${VRAM_USED}" ] && [ -n "${VRAM_TOTAL}" ] && [ "${VRAM_TOTAL}" -gt 0 ] 2>/dev/null; then
  VRAM_PCT=$((VRAM_USED * 100 / VRAM_TOTAL))
  echo "GPU[0] VRAM usage: ${VRAM_PCT}% (${VRAM_USED} / ${VRAM_TOTAL} bytes)"
  if [ "${VRAM_PCT}" -gt 50 ]; then
    echo "WARNING: VRAM occupancy ${VRAM_PCT}% > 50% before sglang start — possible stale process" >&2
  fi
else
  echo "(unable to parse VRAM usage)"
fi
echo ""

# 5. Host Memory
echo "--- Host Memory ---"
free -h | head -3
MEM_AVAIL_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
MEM_TOTAL_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
if [ "${MEM_TOTAL_KB}" -gt 0 ] 2>/dev/null; then
  MEM_PCT=$(( (MEM_TOTAL_KB - MEM_AVAIL_KB) * 100 / MEM_TOTAL_KB ))
  echo "Host memory usage: ${MEM_PCT}% (${MEM_AVAIL_KB} KB available / ${MEM_TOTAL_KB} KB total)"
  if [ "${MEM_PCT}" -gt 80 ]; then
    echo "WARNING: Host memory usage ${MEM_PCT}% > 80%" >&2
  fi
else
  echo "(unable to parse host memory)"
fi
echo ""

# 6. IB Device Status
echo "--- IB Device Status ---"
ibv_devinfo 2>&1 | grep -E "hca_id|transport|fw_ver|state|link_layer" | head -40 || echo "(ibv_devinfo not available)"
IB_ACTIVE=$(ibv_devinfo 2>&1 | grep -c "PORT_ACTIVE" || true)
echo "Active IB ports: ${IB_ACTIVE}"
if [ "${IB_ACTIVE}" -lt 1 ]; then
  echo "WARNING: No active IB ports detected — PD transfer will fall back to TCP" >&2
fi
echo ""

# 7. Model File Check
echo "--- Model File Check ---"
MODEL_PATH="/data/model/glm52-fp8"
if [ ! -d "${MODEL_PATH}" ]; then
  echo "FATAL: Model directory ${MODEL_PATH} not found" >&2
  exit 1
fi
if [ ! -f "${MODEL_PATH}/config.json" ]; then
  echo "FATAL: ${MODEL_PATH}/config.json missing" >&2
  exit 1
fi
echo "Model path: ${MODEL_PATH}"
ls -la "${MODEL_PATH}/config.json"
ls "${MODEL_PATH}/"*.safetensors 2>/dev/null | head -5 || echo "(no .safetensors files found)"
MODEL_SIZE=$(du -sh "${MODEL_PATH}" 2>/dev/null | awk '{print $1}')
echo "Model size: ${MODEL_SIZE}"
echo ""

# 8. GEMM CSV: extract ONLY newly-tuned gfx942 K=6144 entries to a model_configs file
# IMPORTANT:
# - Do NOT replace the image's source CSV — that causes aiter's merge process to
#   truncate all model_configs/*.csv files to 0 rows.
# - Do NOT include ALL gfx942 entries — many already exist in other model_configs
#   (glm5, dsv3, etc.) and duplicates cause RuntimeError on import.
# - ONLY include the shapes we actually tuned: gfx942 with K=6144 (N=256 and N=32).
#   These were missing from all existing configs and caused "not found tuned config"
#   warnings at runtime.
echo "--- aiter GEMM CSV ---"
SRC_CSV="/data/aiter_configs/bf16_tuned_gemm.csv"
OUT_CSV="/shared/mi308x_gfx942_bf16_tuned_gemm.csv"
if [ -f "${SRC_CSV}" ]; then
  # Extract ONLY genuinely new gfx942 K=6144 entries that don't already exist
  # in the image's model_configs files. The aiter merge process raises
  # RuntimeError on duplicate shape keys, so we must filter them out here.
  python3 << 'PYEOF'
import csv, glob, os, sys

src = "/data/aiter_configs/bf16_tuned_gemm.csv"
out = "/shared/mi308x_gfx942_bf16_tuned_gemm.csv"
configs_dir = "/sgl-workspace/aiter/aiter/configs/model_configs"

# Collect existing shape keys (first 10 fields) from all model_configs CSVs
existing_keys = set()
for f in glob.glob(os.path.join(configs_dir, "*.csv")):
    try:
        with open(f, newline='') as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 10:
                    existing_keys.add(tuple(row[:10]))
    except Exception:
        pass

count = 0
total = 0
with open(src, newline='') as fin, open(out, 'w', newline='') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    try:
        header = next(reader)
        writer.writerow(header)
    except StopIteration:
        sys.exit(0)
    for row in reader:
        if len(row) >= 10 and row[0] == "gfx942" and row[4] == "6144":
            total += 1
            if tuple(row[:10]) not in existing_keys:
                writer.writerow(row)
                count += 1

print(f"Extracted {count} new gfx942 K=6144 entries (of {total} total, {total - count} duplicates filtered)")
PYEOF
  TOTAL_ROWS=$(wc -l < "${OUT_CSV}")
  echo "Wrote ${TOTAL_ROWS} lines (including header) to ${OUT_CSV}"
else
  echo "WARNING: ${SRC_CSV} not found — will use image default configs" >&2
  touch "${OUT_CSV}"  # empty sentinel so main container knows
fi
echo ""

# 9. Mooncake patched files check
echo "--- Mooncake Patched Files ---"
MC_DIR="/data/mooncake-patched"
if [ -d "${MC_DIR}" ]; then
  ls -la "${MC_DIR}/" | head -10
else
  echo "WARNING: ${MC_DIR} not found — main container will fail" >&2
fi
echo ""

echo "========== Pre-Flight Check Complete =========="
"""

# Lines to inject into the main container's startup script, right before
# `exec python3 -m sglang.launch_server`.
GEMM_CSV_INSTALL_SNIPPET = """          # Install tuned aiter GEMM configs from initContainer
          # Copy as a model_configs file (NOT replacing the source CSV) to avoid
          # truncating existing model_configs during aiter's merge process.
          if [ -s /shared/mi308x_gfx942_bf16_tuned_gemm.csv ]; then
            cp /shared/mi308x_gfx942_bf16_tuned_gemm.csv \
               /sgl-workspace/aiter/aiter/configs/model_configs/mi308x_gfx942_bf16_tuned_gemm.csv
            echo "Installed tuned GEMM configs to model_configs/mi308x_gfx942_bf16_tuned_gemm.csv"
          else
            echo "No tuned GEMM configs from initContainer — using image defaults"
          fi
"""


def kubectl_json(args: list[str]) -> dict:
    """Run kubectl and return parsed JSON output."""
    result = subprocess.run(
        ["kubectl", "get", "-n", NAMESPACE] + args + ["-o", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def kubectl_apply(manifest: dict) -> str:
    """Apply a manifest via kubectl apply."""
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=json.dumps(manifest),
        capture_output=True,
        text=True,
    )
    return f"{result.stdout.strip()}{result.stderr.strip()}"


def build_init_container(image: str) -> dict:
    """Build the initContainer spec."""
    return {
        "name": INIT_CONTAINER_NAME,
        "image": image,
        "imagePullPolicy": "Always",
        "command": ["/bin/bash", "-c"],
        "args": [INIT_SCRIPT],
        "securityContext": {
            "allowPrivilegeEscalation": True,
            "privileged": True,
            "readOnlyRootFilesystem": False,
            "seccompProfile": {"type": "Unconfined"},
        },
        "volumeMounts": [
            {"mountPath": "/data", "name": "data"},
            {"mountPath": SHARED_VOLUME_MOUNT, "name": SHARED_VOLUME_NAME},
            {"mountPath": "/dev/kfd", "name": "dev-kfd"},
            {"mountPath": "/dev/dri", "name": "dev-dri"},
            {"mountPath": "/dev/infiniband", "name": "dev-infiniband"},
        ],
        "resources": {
            "limits": {"amd.com/gpu": "8"},
            "requests": {"cpu": "4", "memory": "8Gi"},
        },
    }


def build_shared_volume() -> dict:
    """Build the shared emptyDir volume spec."""
    return {
        "name": SHARED_VOLUME_NAME,
        "emptyDir": {"medium": "Memory", "sizeLimit": "16Mi"},
    }


def patch_main_container_args(args: str) -> str:
    """Inject GEMM CSV install snippet into main container startup script.

    The snippet is inserted right before `exec python3 -m sglang.launch_server`.
    If already patched (idempotency check), returns unchanged.
    """
    marker = "Install tuned aiter GEMM configs from initContainer"
    if marker in args:
        return args  # already patched

    # Find the exec line and insert before it
    exec_line = "exec python3 -m sglang.launch_server"
    if exec_line not in args:
        print("WARNING: Could not find 'exec python3 -m sglang.launch_server' in args")
        return args

    return args.replace(exec_line, GEMM_CSV_INSTALL_SNIPPET + "          " + exec_line)


def patch_sts(sts_name: str, dry_run: bool = False) -> bool:
    """Patch a StatefulSet to add initContainer, shared volume, and main container modifications.

    Returns True if changes were made (or would be made in dry-run).
    """
    print(f"\n{'=' * 60}")
    print(f"Patching StatefulSet: {sts_name}")
    print(f"{'=' * 60}")

    # Fetch current StatefulSet
    sts = kubectl_json(["sts", sts_name])
    spec = sts["spec"]["template"]["spec"]
    containers = spec.get("containers", [])
    volumes = spec.get("volumes", [])

    if not containers:
        print(f"ERROR: No containers found in {sts_name}")
        return False

    main_container = containers[0]
    image = main_container["image"]
    print(f"Main container image: {image}")

    changes_made = []

    # 1. Add initContainer (if not present)
    init_containers = spec.get("initContainers", [])
    existing_init_names = [c["name"] for c in init_containers]

    if INIT_CONTAINER_NAME not in existing_init_names:
        init_container = build_init_container(image)
        init_containers.append(init_container)
        spec["initContainers"] = init_containers
        changes_made.append(f"Added initContainer '{INIT_CONTAINER_NAME}'")
    else:
        # Update existing initContainer
        for i, c in enumerate(init_containers):
            if c["name"] == INIT_CONTAINER_NAME:
                init_containers[i] = build_init_container(image)
                changes_made.append(f"Updated existing initContainer '{INIT_CONTAINER_NAME}'")
                break

    # 2. Add shared volume (if not present)
    volume_names = [v["name"] for v in volumes]
    if SHARED_VOLUME_NAME not in volume_names:
        volumes.append(build_shared_volume())
        spec["volumes"] = volumes
        changes_made.append(f"Added shared volume '{SHARED_VOLUME_NAME}'")
    else:
        # Update existing volume
        for i, v in enumerate(volumes):
            if v["name"] == SHARED_VOLUME_NAME:
                volumes[i] = build_shared_volume()
                changes_made.append(f"Updated existing volume '{SHARED_VOLUME_NAME}'")
                break

    # 3. Add shared volumeMount to main container (if not present)
    volume_mounts = main_container.get("volumeMounts", [])
    mount_paths = [vm["mountPath"] for vm in volume_mounts]

    if SHARED_VOLUME_MOUNT not in mount_paths:
        volume_mounts.append({"mountPath": SHARED_VOLUME_MOUNT, "name": SHARED_VOLUME_NAME})
        main_container["volumeMounts"] = volume_mounts
        changes_made.append(f"Added volumeMount '{SHARED_VOLUME_MOUNT}' to main container")
    else:
        changes_made.append(f"volumeMount '{SHARED_VOLUME_MOUNT}' already present")

    # 4. Patch main container args to install GEMM CSV
    args_list = main_container.get("args", [])
    if args_list:
        original_args = args_list[0]
        patched_args = patch_main_container_args(original_args)
        if patched_args != original_args:
            args_list[0] = patched_args
            main_container["args"] = args_list
            changes_made.append("Injected GEMM CSV install snippet into main container startup")
        else:
            changes_made.append("GEMM CSV install snippet already present (idempotent)")

    # Write back containers (in case of structural changes)
    spec["containers"][0] = main_container

    if not changes_made:
        print("No changes needed — already patched.")
        return False

    print(f"\nChanges ({len(changes_made)}):")
    for c in changes_made:
        print(f"  + {c}")

    if dry_run:
        print("\n[DRY RUN] Would apply the above changes.")
        return True

    # Apply the patched StatefulSet
    # Strip status and metadata fields that shouldn't be in apply input
    apply_manifest = {
        "apiVersion": sts["apiVersion"],
        "kind": sts["kind"],
        "metadata": {
            "name": sts["metadata"]["name"],
            "namespace": sts["metadata"]["namespace"],
            "labels": sts["metadata"].get("labels", {}),
            "annotations": sts["metadata"].get("annotations", {}),
        },
        "spec": sts["spec"],
    }

    result = kubectl_apply(apply_manifest)
    print(f"\nApply result: {result}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add initContainer with env checks + GEMM CSV persistence to 1p1d PD StatefulSets"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    parser.add_argument("--prefill", action="store_true", help="Only patch prefill StatefulSet")
    parser.add_argument("--decode", action="store_true", help="Only patch decode StatefulSet")
    args = parser.parse_args()

    # Default: patch both
    targets = []
    if args.prefill or (not args.prefill and not args.decode):
        targets.append(f"{STS_PREFIX}-prefill")
    if args.decode or (not args.prefill and not args.decode):
        targets.append(f"{STS_PREFIX}-decode")

    print(f"Target StatefulSets: {', '.join(targets)}")
    print(f"Dry run: {args.dry_run}")

    any_changes = False
    for sts_name in targets:
        try:
            changed = patch_sts(sts_name, dry_run=args.dry_run)
            any_changes = any_changes or changed
        except subprocess.CalledProcessError as e:
            print(f"ERROR patching {sts_name}: {e}")
            print(f"  stderr: {e.stderr}" if e.stderr else "")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR patching {sts_name}: {e}")
            sys.exit(1)

    if any_changes and not args.dry_run:
        print(f"\n{'=' * 60}")
        print("All patches applied successfully.")
        print("\nNext steps:")
        print("  1. Verify: kubectl get sts -n kube-system sglang-1p1d-prefill -o yaml | grep -A5 initContainers")
        print("  2. Restart: kubectl rollout restart sts -n kube-system sglang-1p1d-prefill")
        print("  3. Restart: kubectl rollout restart sts -n kube-system sglang-1p1d-decode")
        print("  4. Check init logs: kubectl logs -n kube-system sglang-1p1d-prefill-0 -c env-check-aiter-init")
        print("  5. Verify CSV: kubectl logs -n kube-system sglang-1p1d-prefill-0 | grep 'Installed tuned GEMM'")
    elif not any_changes:
        print("\nNo changes were necessary — StatefulSets already patched.")


if __name__ == "__main__":
    main()
