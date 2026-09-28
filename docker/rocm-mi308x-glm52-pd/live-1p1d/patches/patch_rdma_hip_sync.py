#!/usr/bin/env python3
"""
Patch mooncake/conn.py MooncakeKVReceiver.poll() to add GPU memory barrier
on AMD HIP after RDMA KV transfer completion.

Root cause: On AMD HIP/ROCm, RDMA writes to GPU VRAM via GDR/dmabuf are NOT
automatically coherent with the GPU L2 cache. The ZMQ "Success" message from
prefill arrives before the GPU L2 cache invalidates stale cache lines, causing
decode to read partially stale KV data → garbled output on cold-cache runs.

Fix: When poll() returns Success on HIP, insert torch.cuda.synchronize()
(which maps to hipDeviceSynchronize on ROCm) to ensure all RDMA writes are
visible to the GPU compute units.
"""
from pathlib import Path

TARGET = Path("/sgl-workspace/sglang/python/sglang/srt/disaggregation/mooncake/conn.py")

# MooncakeKVReceiver.poll() at line ~2411 in v0.5.17
# This is the decode-side receiver's poll method.
OLD = """    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
        elif status == KVPoll.WaitingForInput:
            timeout_result = self._check_waiting_timeout()
            if timeout_result is not None:
                return timeout_result

        return status"""

NEW = """    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            # AMD HIP: RDMA writes via GDR/dmabuf are NOT automatically coherent
            # with GPU L2 cache. The ZMQ "Success" message from prefill can arrive
            # before the GPU L2 cache invalidates stale cache lines. Without this
            # sync, decode reads partially stale KV data → garbled cold-cache output.
            # On CUDA, GPUDirect RDMA writes ARE coherent, so this is a no-op there.
            if status == KVPoll.Success:
                import torch
                if torch.version.hip is not None:
                    torch.cuda.synchronize()
            self.conclude_state = status
        elif status == KVPoll.WaitingForInput:
            timeout_result = self._check_waiting_timeout()
            if timeout_result is not None:
                return timeout_result

        return status"""

text = TARGET.read_text()
old_count = text.count(OLD)

if old_count == 1:
    text = text.replace(OLD, NEW, 1)
    TARGET.write_text(text)
    status = "applied"
elif old_count == 0:
    if "RDMA writes via GDR/dmabuf are NOT automatically coherent" in text:
        status = "already-patched"
    else:
        # Try to find the poll method to debug
        import re
        polls = [(m.start(), m.group()) for m in re.finditer(r'def poll\(self\).*?return status', text, re.DOTALL)]
        print(f"DEBUG: found {len(polls)} poll methods")
        for i, (pos, match) in enumerate(polls):
            print(f"  poll {i} at char {pos}: {match[:100]}...")
        raise RuntimeError(
            "OLD pattern not found and patch not already applied — "
            "conn.py may have changed. See debug output above."
        )
else:
    raise RuntimeError(f"Unexpected: OLD pattern found {old_count} times")

# Verify
verified = TARGET.read_text()
if status == "applied":
    assert NEW in verified, "Patch verification failed: NEW not found after write"
    assert verified.count(OLD) == 0, "Patch verification failed: OLD still present"

print(f"PATCH_RDMA_HIP_SYNC={status}")
