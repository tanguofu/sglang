#!/bin/bash
set -euo pipefail

# Entrypoint for native SGLang router (no Python proxy).
# Installs patched wheel from hostPath cache if available, then starts router.

WHEEL_CACHE="/wheel-cache/sglang_router-0.3.2-cp38-abi3-manylinux_2_34_x86_64.whl"

if [ -f "${WHEEL_CACHE}" ]; then
  echo "=== Installing patched router wheel ==="
  pip install --force-reinstall --no-deps "${WHEEL_CACHE}" 2>&1 | tail -3
  echo "=== Wheel installed successfully ==="
else
  echo "=== WARNING: Patched wheel not found at ${WHEEL_CACHE}, using base image router ==="
fi

echo "=== Starting SGLang Router (native, no proxy) ==="
exec python3 -m sglang_router.launch_router "$@"
