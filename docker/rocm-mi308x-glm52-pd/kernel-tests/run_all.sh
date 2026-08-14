#!/usr/bin/env bash
# Run gfx942 kernel unit tests. KERNEL_TEST_SHARD=a|b|all
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
SHARD="${KERNEL_TEST_SHARD:-all}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

case "${SHARD}" in
  a|A)
    files=(test_fp8_mqa_logits.py test_fused_store_index_cache.py test_rmsnorm.py)
    ;;
  b|B)
    files=(test_aiter_gemm_bf16.py test_aiter_fmoe.py test_rope.py)
    ;;
  all|*)
    files=(
      test_fp8_mqa_logits.py
      test_fused_store_index_cache.py
      test_rmsnorm.py
      test_aiter_gemm_bf16.py
      test_aiter_fmoe.py
      test_rope.py
    )
    ;;
esac

python3 -m pip install -q pytest
echo "[kernel-tests] shard=${SHARD} files=${files[*]} device=${HIP_VISIBLE_DEVICES}"
cd "${DIR}"
exec python3 -m pytest -v --tb=short "${files[@]}"
