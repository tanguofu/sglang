#!/usr/bin/env python3
"""Inject sender-side hipMemcpy D2H into mooncake _transfer_data.

Prefill RDMA-reads GPU VRAM. On HIP without peermem, that read is not L2
coherent with a just-computed KV cache, so true cold PD transfers send
garbage. Copy each transfer block GPU→host first, then RDMA from host.

Does not replace kv_data_ptrs or change send() signature.
Idempotent.
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

def _run_gdr_flush():
    gdr = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patch_gdr_flush.py")
    if os.path.exists(gdr):
        os.system(f"python3 {gdr} {CONN_PATH}")


if "FIX(prefill-d2h-host-staging)" in src:
    print("patch_prefill_d2h: already patched, skipping")
    _run_gdr_flush()
    sys.exit(0)

OLD = """    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        if not transfer_blocks:
            return 0

        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )
"""

NEW = '''    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        if not transfer_blocks:
            return 0

        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        # FIX(prefill-d2h-host-staging): HIP RDMA read of GPU VRAM is not
        # L2-coherent after a fresh prefill. Copy blocks to host first.
        if os.environ.get("SGLANG_PD_HOST_STAGING") == "1":
            import ctypes

            hip_lib = ctypes.CDLL("libamdhip64.so")
            gpu_id = getattr(self.kv_args, "gpu_id", 0)
            hip_lib.hipSetDevice(ctypes.c_int(gpu_id))
            host_cptrs = []
            host_addrs = []
            host_lens = []
            for src_addr, _dst, length in transfer_blocks:
                host_ptr = ctypes.c_void_p()
                alloc_ret = hip_lib.hipMallocHost(
                    ctypes.byref(host_ptr), ctypes.c_size_t(length)
                )
                if alloc_ret != 0:
                    logger.error(
                        f"_transfer_data: hipMallocHost failed ret={alloc_ret} "
                        f"len={length} dev={gpu_id}"
                    )
                    for cp in host_cptrs:
                        hip_lib.hipFreeHost(cp)
                    return alloc_ret
                cp_ret = hip_lib.hipMemcpy(
                    host_ptr,
                    ctypes.c_void_p(int(src_addr)),
                    ctypes.c_size_t(length),
                    ctypes.c_int(2),  # hipMemcpyDeviceToHost
                )
                if cp_ret != 0:
                    logger.error(
                        f"_transfer_data: D2H failed ret={cp_ret} "
                        f"src=0x{int(src_addr):x} len={length} dev={gpu_id}"
                    )
                    hip_lib.hipFreeHost(host_ptr)
                    for cp in host_cptrs:
                        hip_lib.hipFreeHost(cp)
                    return cp_ret
                host_cptrs.append(host_ptr)
                host_addrs.append(host_ptr.value)
                host_lens.append(length)
            logger.info(
                f"_transfer_data: D2H {len(host_addrs)} blocks "
                f"({sum(host_lens)} bytes) then RDMA dev={gpu_id}"
            )
            try:
                self.engine.batch_register(host_addrs, host_lens)
                ret = self.engine.batch_transfer_sync(
                    mooncake_session_id, host_addrs, list(dst_addrs), list(lengths)
                )
            finally:
                try:
                    self.engine.batch_deregister(host_addrs)
                except Exception:
                    pass
                for cp in host_cptrs:
                    hip_lib.hipFreeHost(cp)
            return ret

        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )
'''

if OLD not in src:
    print("patch_prefill_d2h: WARNING - _transfer_data anchor not found")
    sys.exit(1)

src = src.replace(OLD, NEW, 1)
tmp = CONN_PATH + ".d2hpatched"
with open(tmp, "w") as f:
    f.write(src)
try:
    py_compile.compile(tmp, doraise=True)
    os.rename(tmp, CONN_PATH)
    print(f"patch_prefill_d2h: SUCCESS - patched {CONN_PATH}")
    _run_gdr_flush()
except py_compile.PyCompileError as e:
    os.unlink(tmp)
    print(f"patch_prefill_d2h: FAILED - {e}")
    sys.exit(1)
