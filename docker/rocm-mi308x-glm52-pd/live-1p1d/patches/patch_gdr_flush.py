#!/usr/bin/env python3
"""Inject cheap GDR L2/PCIe flush into mooncake conn.py.

When SGLANG_PD_HOST_STAGING!=1:
  prefill _transfer_data:
    1) GPU L2 writeback (buffer_wbl2) so NIC GDR reads fresh KV
    2) RDMA WRITE
    3) 8-byte RDMA READ of the last dest addr (NCCL-style) so decode
       posted writes commit before ZMQ Success
  decode poll Success:
    GPU L2 invalidate (buffer_inv) so decode kernels refetch HBM

Idempotent. Does not enable host-staging bounce buffers.
"""
import os
import py_compile
import sys

CONN_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/sgl-workspace/sglang/python/sglang/srt/disaggregation/mooncake/conn.py"
)

with open(CONN_PATH) as f:
    src = f.read()

# Transfer and poll are independent. v0.5.17 poll() no longer contains the
# host-staging copy block, so an earlier run could patch transfer (set the
# FIX marker) and skip poll forever.
transfer_done = "FIX(gdr-l2-flush)" in src and "rdma_read_flush" in src
poll_done = "poll: GDR L2 invalidate" in src
if transfer_done and poll_done:
    print("patch_gdr_flush: already patched, skipping")
    sys.exit(0)

OLD_XFER = """        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )
"""

NEW_XFER = '''        # FIX(gdr-l2-flush): cheap coherence for true GDR (no 23GB D2H).
        if os.environ.get("SGLANG_PD_HOST_STAGING") != "1":
            try:
                from sglang.srt.disaggregation.mooncake.gdr_l2_flush import (
                    ensure_read_sink,
                    rdma_read_flush,
                    writeback,
                )
            except ImportError:
                import sys as _sys
                if "/data/mooncake-patched" not in _sys.path:
                    _sys.path.insert(0, "/data/mooncake-patched")
                from gdr_l2_flush import ensure_read_sink, rdma_read_flush, writeback

            gpu_id = getattr(self.kv_args, "gpu_id", 0)
            if not writeback(gpu_id):
                logger.error("_transfer_data: GDR L2 writeback failed dev=%s", gpu_id)
            ret = self.engine.batch_transfer_sync(
                mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
            )
            if ret == 0 and dst_addrs and lengths:
                sink = ensure_read_sink(self.engine, gpu_id)
                last_len = int(lengths[-1])
                flush_src = int(dst_addrs[-1]) + max(last_len, 8) - 8
                flush_src &= ~7
                if sink:
                    try:
                        flush_ret = rdma_read_flush(
                            self.engine, mooncake_session_id, sink, flush_src, 8
                        )
                    except Exception as e:
                        logger.error(
                            "_transfer_data: GDR RDMA READ flush raised: %s", e
                        )
                        flush_ret = -1
                    if flush_ret != 0:
                        logger.error(
                            "_transfer_data: GDR RDMA READ flush ret=%s "
                            "dst=0x%x dev=%s",
                            flush_ret,
                            flush_src,
                            gpu_id,
                        )
                    else:
                        logger.info(
                            "_transfer_data: GDR flush wb+READ 8B dst=0x%x "
                            "blocks=%s bytes=%s dev=%s",
                            flush_src,
                            len(lengths),
                            sum(int(x) for x in lengths),
                            gpu_id,
                        )
            return ret

        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )
'''

# v0.5.17 image poll() (staging-outstanding hold, no host-staging copy).
OLD_POLL_V0517 = """            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
                self.trace_ctx.trace_req_finish()
"""

NEW_POLL_V0517 = """            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
                self.trace_ctx.trace_req_finish()
                if (
                    status == KVPoll.Success
                    and os.environ.get("SGLANG_PD_HOST_STAGING") != "1"
                ):
                    # FIX(gdr-l2-flush): invalidate decode L2 after GDR WRITE.
                    try:
                        from sglang.srt.disaggregation.mooncake.gdr_l2_flush import (
                            invalidate,
                        )
                    except ImportError:
                        import sys as _sys
                        if "/data/mooncake-patched" not in _sys.path:
                            _sys.path.insert(0, "/data/mooncake-patched")
                        from gdr_l2_flush import invalidate

                    gpu_id = getattr(self.kv_mgr.kv_args, "gpu_id", 0)
                    if not invalidate(gpu_id):
                        logger.error(
                            "poll: GDR L2 invalidate failed room=%s dev=%s",
                            self.bootstrap_room,
                            gpu_id,
                        )
"""

# Older host-staging poll copy block.
OLD_POLL_STAGING = """            if (
                status == KVPoll.Success
                and hasattr(self.kv_mgr, "_host_staging_buffers")
            ):
                self.kv_mgr._copy_host_to_gpu(getattr(self, "_dst_kv_indices", None))
"""

NEW_POLL_STAGING = """            if (
                status == KVPoll.Success
                and hasattr(self.kv_mgr, "_host_staging_buffers")
            ):
                self.kv_mgr._copy_host_to_gpu(getattr(self, "_dst_kv_indices", None))
            elif (
                status == KVPoll.Success
                and os.environ.get("SGLANG_PD_HOST_STAGING") != "1"
            ):
                # FIX(gdr-l2-flush): invalidate decode L2 after GDR WRITE.
                try:
                    from sglang.srt.disaggregation.mooncake.gdr_l2_flush import (
                        invalidate,
                    )
                except ImportError:
                    import sys as _sys
                    if "/data/mooncake-patched" not in _sys.path:
                        _sys.path.insert(0, "/data/mooncake-patched")
                    from gdr_l2_flush import invalidate

                gpu_id = getattr(self.kv_mgr.kv_args, "gpu_id", 0)
                if not invalidate(gpu_id):
                    logger.error(
                        "poll: GDR L2 invalidate failed room=%s dev=%s",
                        self.bootstrap_room,
                        gpu_id,
                    )
"""

changed = []
if transfer_done:
    print("patch_gdr_flush: transfer already patched")
elif OLD_XFER not in src:
    print("patch_gdr_flush: WARNING - _transfer_data return anchor not found")
else:
    src = src.replace(OLD_XFER, NEW_XFER, 1)
    changed.append("transfer")

if poll_done:
    print("patch_gdr_flush: poll already patched")
elif OLD_POLL_V0517 in src:
    src = src.replace(OLD_POLL_V0517, NEW_POLL_V0517, 1)
    changed.append("poll-v0517")
elif OLD_POLL_STAGING in src:
    src = src.replace(OLD_POLL_STAGING, NEW_POLL_STAGING, 1)
    changed.append("poll-staging")
else:
    print("patch_gdr_flush: WARNING - decode poll anchor not found")

if not changed:
    print("patch_gdr_flush: NO CHANGES")
    sys.exit(1)

tmp = CONN_PATH + ".gdrflush"
with open(tmp, "w") as f:
    f.write(src)
try:
    py_compile.compile(tmp, doraise=True)
    os.rename(tmp, CONN_PATH)
    print(f"patch_gdr_flush: SUCCESS - {', '.join(changed)} patched {CONN_PATH}")
except py_compile.PyCompileError as e:
    os.unlink(tmp)
    print(f"patch_gdr_flush: FAILED - {e}")
    sys.exit(1)
