#!/bin/bash
for c in sglang_pd_stack sglang_pd_prefill sglang_pd_decode sglang_pd_router sglang_glm52_tp8mtp sglang_plan_b sglang_perf_pd_decode; do
  docker rm -f "$c" 2>/dev/null || true
done
echo "All PD/unified containers stopped"
