#!/usr/bin/env python3
"""
P2: Patch is_fully_idle(for_health_check=True) to also check disagg queues.

Root cause: when prealloc/transfer queue has items but running_batch is empty,
is_fully_idle returns True for health checks → health check request enters the
normal processing path → process_decode_queue() blocks 15s+ on synchronous
handshake/gloo collective → exceeds 20s (now 60s) health check timeout → 503.

Fix: for_health_check=True时，disagg队列非空 → 视为"not idle"(忙=健康) →
跳过健康检查，避免阻塞在process_decode_queue的同步路径上。

v2: Only apply to DECODE mode. Prefill's bootstrap/inflight queues are network
    operations, not GPU work — checking them causes false "not idle" and skips
    health checks when the server is actually idle → 503.
"""
import sys

FILE = "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py"

try:
    with open(FILE, "r") as f:
        content = f.read()
except FileNotFoundError:
    print(f"ERROR: {FILE} not found", file=sys.stderr)
    sys.exit(1)

MARKER = "# HEALTH_CHECK_PATCH: disagg queues v2"
if MARKER in content:
    print(f"scheduler.py already patched ({MARKER}), skipping")
    sys.exit(0)

# If v1 was applied, revert it first
V1_MARKER = "# HEALTH_CHECK_PATCH: disagg queues v1"
if V1_MARKER in content:
    # Find and remove the v1 patch block
    lines = content.split("\n")
    new_lines = []
    skip = False
    for i, line in enumerate(lines):
        if V1_MARKER in line:
            skip = True
            continue
        if skip:
            # Skip until we find the end of the v1 block (the else: before the marker)
            # v1 block ends with the last idle &= line
            if line.strip().startswith("idle &=") and "disagg_prefill_bootstrap_queue" in line:
                skip = False
                continue  # skip this line too
            continue
        # Also remove the "else:" that was added before the v1 marker
        if (i > 0 and "HEALTH_CHECK_PATCH" in lines[i-1] and
            line.strip() == "else:" and
            any("disagg queues v1" in lines[j] for j in range(max(0, i-5), i))):
            continue
        new_lines.append(line)
    content = "\n".join(new_lines)
    print("Reverted v1 patch")

OLD = """        if not for_health_check:
            # Grammar queue and prefill inflight queue may not produce batch
            # results instantly, but they still indicate the server is not idle.
            idle &= len(self.grammar_manager.grammar_queue) == 0
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                idle &= len(self.disagg_prefill_inflight_queue) == 0
                idle &= len(self.disagg_prefill_bootstrap_queue.queue) == 0

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                idle &= len(self.disagg_decode_prealloc_queue.queue) == 0
                idle &= len(self.disagg_decode_prealloc_queue.retracted_queue) == 0
                idle &= len(self.disagg_decode_transfer_queue.queue) == 0
                if self.decode_offload_manager is not None:
                    idle &= len(self.decode_offload_manager.ongoing_offload) == 0"""

NEW = """        if not for_health_check:
            # Grammar queue and prefill inflight queue may not produce batch
            # results instantly, but they still indicate the server is not idle.
            idle &= len(self.grammar_manager.grammar_queue) == 0
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                idle &= len(self.disagg_prefill_inflight_queue) == 0
                idle &= len(self.disagg_prefill_bootstrap_queue.queue) == 0

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                idle &= len(self.disagg_decode_prealloc_queue.queue) == 0
                idle &= len(self.disagg_decode_prealloc_queue.retracted_queue) == 0
                idle &= len(self.disagg_decode_transfer_queue.queue) == 0
                if self.decode_offload_manager is not None:
                    idle &= len(self.decode_offload_manager.ongoing_offload) == 0
        else:
            # HEALTH_CHECK_PATCH: disagg queues v2
            # for_health_check=True: treat non-empty disagg queues as "not idle"
            # (busy = healthy) ONLY for DECODE mode. Decode's prealloc/transfer
            # queues indicate active handshake/transfer work — the server is alive
            # and processing, so skip health check to avoid blocking on
            # process_decode_queue's synchronous handshake/collective ops.
            # PREFILL mode is NOT patched: bootstrap/inflight queues are network
            # operations that don't block the GPU — checking them would falsely
            # skip health checks when the server is actually idle.
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if self.disagg_decode_prealloc_queue is not None:
                    idle &= len(self.disagg_decode_prealloc_queue.queue) == 0
                    idle &= len(self.disagg_decode_transfer_queue.queue) == 0"""

if OLD not in content:
    print("ERROR: could not find target code block in scheduler.py", file=sys.stderr)
    for line in OLD.split("\n")[:3]:
        if line.strip() and line.strip() not in content:
            print(f"  missing line: {line.strip()!r}", file=sys.stderr)
    sys.exit(1)

content = content.replace(OLD, NEW, 1)

with open(FILE, "w") as f:
    f.write(content)

print(f"OK: Applied health check patch v2 to {FILE} (decode-only disagg check)")
