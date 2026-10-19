#!/bin/bash
# init-env-check.sh — initContainer script for 1p1d PD StatefulSets
#
# Runs environment pre-flight checks (driver, firmware, VRAM, memory, IB, model)
# and copies tuned aiter GEMM CSV from /data/aiter_configs/ to /shared/ for
# the main container to install before starting sglang.
#
# Exit codes:
#   0 = all checks passed (warnings are non-fatal)
#   1 = FATAL: critical check failed (no GPU, no model, config missing)

set -euo pipefail

echo "========== Environment Pre-Flight Check =========="
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Hostname:  $(hostname)"
echo ""

# ---------------------------------------------------------------------------
# 1. GPU Detection
# ---------------------------------------------------------------------------
echo "--- GPU Detection ---"
GPU_COUNT=$(rocminfo 2>/dev/null | grep -c "^  Name:.*gfx" || true)
echo "GPU count: ${GPU_COUNT}"
if [ "${GPU_COUNT}" -lt 1 ]; then
  echo "FATAL: No GPU detected via rocminfo" >&2
  exit 1
fi
rocminfo 2>/dev/null | grep -E "Marketing Name:|Name:.*gfx|Compute Unit:" | head -12
echo ""

# ---------------------------------------------------------------------------
# 2. Driver Version
# ---------------------------------------------------------------------------
echo "--- Driver Version ---"
rocm-smi --showdriverversion 2>&1 | grep -i "driver version" || echo "(unable to query driver version)"
echo ""

# ---------------------------------------------------------------------------
# 3. Firmware Version (GPU 0 only for brevity)
# ---------------------------------------------------------------------------
echo "--- Firmware Version (GPU 0) ---"
rocm-smi --showfwinfo 2>&1 | grep "GPU\[0\]" | head -10 || echo "(unable to query firmware info)"
echo ""

# ---------------------------------------------------------------------------
# 4. VRAM Occupancy (pre-sglang, should be near-zero)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 5. Host Memory
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 6. IB Device Status (critical for PD mooncake RDMA transfer)
# ---------------------------------------------------------------------------
echo "--- IB Device Status ---"
ibv_devinfo 2>&1 | grep -E "hca_id|transport|fw_ver|state|link_layer" | head -40 || echo "(ibv_devinfo not available)"
IB_ACTIVE=$(ibv_devinfo 2>&1 | grep -c "PORT_ACTIVE" || true)
echo "Active IB ports: ${IB_ACTIVE}"
if [ "${IB_ACTIVE}" -lt 1 ]; then
  echo "WARNING: No active IB ports detected — PD transfer will fall back to TCP" >&2
fi
echo ""

# ---------------------------------------------------------------------------
# 7. Model File Check
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 8. GEMM CSV: extract ONLY newly-tuned gfx942 K=6144 entries to a model_configs file
# IMPORTANT:
# - Do NOT replace the image's source CSV — that causes aiter's merge process to
#   truncate all model_configs/*.csv files to 0 rows.
# - Do NOT include ALL gfx942 entries — many already exist in other model_configs
#   (glm5, dsv3, etc.) and duplicates cause RuntimeError on import.
# - ONLY include the shapes we actually tuned: gfx942 with K=6144 (N=256 and N=32).
#   These were missing from all existing configs and caused "not found tuned config"
#   warnings at runtime.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 9. Mooncake patched files check
# ---------------------------------------------------------------------------
echo "--- Mooncake Patched Files ---"
MC_DIR="/data/mooncake-patched"
if [ -d "${MC_DIR}" ]; then
  ls -la "${MC_DIR}/" | head -10
else
  echo "WARNING: ${MC_DIR} not found — main container will fail" >&2
fi
echo ""

echo "========== Pre-Flight Check Complete =========="
