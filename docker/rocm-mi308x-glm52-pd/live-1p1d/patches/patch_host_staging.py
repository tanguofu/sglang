#!/usr/bin/env python3
"""Patch mooncake/conn.py to add SGLANG_PD_HOST_STAGING support.

This patch adds host staging to MooncakeKVManager:
1. register_buffer_to_engine(): allocate host buffers via hipMallocHost,
   register them with the RDMA engine instead of GPU buffers.
2. _copy_host_to_gpu(): copy host buffers back to GPU via hipMemcpy H2D.
3. MooncakeKVReceiver.poll(): call _copy_host_to_gpu() on KV transfer Success.

This fixes cold-cache garbled output on AMD HIP where RDMA writes to GPU VRAM
via GDR/dmabuf are NOT coherent with the GPU L2 cache. Host staging routes
RDMA writes through host RAM (CPU-coherent), then hipMemcpy H2D provides the
necessary GPU memory barrier.

Idempotent: skips if already patched.
"""
import os
import re
import sys
import py_compile

CONN_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/sgl-workspace/sglang/python/sglang/srt/disaggregation/mooncake/conn.py"
)

# --- 1. Read the source ---
with open(CONN_PATH, "r") as f:
    src = f.read()

original_src = src
patched = False

# --- Check if already patched ---
if "_copy_host_to_gpu" in src and "_host_staging_buffers" in src:
    print("patch_host_staging: already patched, skipping")
    sys.exit(0)

# --- 2. Patch register_buffer_to_engine() ---
# The container's version has:
#     def register_buffer_to_engine(self):
#         # Batch register KV data buffers
#         if self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
#             self.engine.batch_register(
#                 self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
#             )
#
# We replace the first if block with host staging logic.

OLD_REGISTER = """    def register_buffer_to_engine(self):
        # Batch register KV data buffers
        if self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
            self.engine.batch_register(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            )"""

NEW_REGISTER = """    def register_buffer_to_engine(self):
        # Decode-only host staging. Prefill keeps GPU kv_data_ptrs;
        # D2H is done in _transfer_data (FIX(prefill-d2h-host-staging)).
        _mode = getattr(self.disaggregation_mode, "value", self.disaggregation_mode)
        _is_decode = str(_mode).lower() == "decode"
        if os.environ.get("SGLANG_PD_HOST_STAGING") == "1" and _is_decode:
            import ctypes

            hip_lib = ctypes.CDLL("libamdhip64.so")
            self._host_staging_buffers = []
            self._host_staging_ptrs = []
            self._host_staging_lens = []
            self._gpu_ptrs = []
            for ptr, length in zip(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            ):
                host_ptr = ctypes.c_void_p()
                alloc_ret = hip_lib.hipMallocHost(
                    ctypes.byref(host_ptr), ctypes.c_size_t(length)
                )
                if alloc_ret != 0:
                    logger.error(f"hipMallocHost failed: ret={alloc_ret}, len={length}")
                    host_ptr = ctypes.c_void_p(ptr)
                self._host_staging_buffers.append(host_ptr)
                self._host_staging_ptrs.append(host_ptr.value)
                self._host_staging_lens.append(length)
                self._gpu_ptrs.append(ptr)
            if self._host_staging_ptrs:
                self.engine.batch_register(
                    self._host_staging_ptrs, self._host_staging_lens
                )
                self.kv_args.kv_data_ptrs = list(self._host_staging_ptrs)
                self._gpu_to_host_map = {}
                for gpu_ptr, host_ptr, length in zip(
                    self._gpu_ptrs, self._host_staging_ptrs, self._host_staging_lens
                ):
                    self._gpu_to_host_map[gpu_ptr] = (host_ptr, length)
                logger.info(
                    f"Host staging: registered {len(self._host_staging_ptrs)} "
                    f"host buffers for KV data (total "
                    f"{sum(self._host_staging_lens)} bytes), "
                    f"replaced kv_data_ptrs with host addresses"
                )
        elif self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
            self.engine.batch_register(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            )"""

if OLD_REGISTER in src:
    src = src.replace(OLD_REGISTER, NEW_REGISTER, 1)
    patched = True
    print("patch_host_staging: patched register_buffer_to_engine()")
else:
    print("patch_host_staging: WARNING - register_buffer_to_engine anchor not found")

# --- 3. Add _copy_host_to_gpu() method after deregister_buffer_to_engine() ---
# Find deregister_buffer_to_engine and add _copy_host_to_gpu after it.

COPY_METHOD = '''
    def _copy_host_to_gpu(self):
        """Copy KV data from host staging buffers to GPU after PD transfer.

        After hipMemcpy (DMA), the RDMA NIC's mapping of host staging buffers
        may become stale, causing 'local access violation' on the next transfer.
        Re-registering the buffers refreshes the NIC's mapping.
        """
        import ctypes

        hip_lib = ctypes.CDLL("libamdhip64.so")
        for host_ptr, gpu_ptr, length in zip(
            self._host_staging_ptrs,
            self._gpu_ptrs,
            self._host_staging_lens,
        ):
            ret = hip_lib.hipMemcpy(
                ctypes.c_void_p(int(gpu_ptr)),
                ctypes.c_void_p(int(host_ptr)),
                ctypes.c_size_t(length),
                ctypes.c_int(1),  # hipMemcpyHostToDevice
            )
            if ret != 0:
                logger.error(
                    f"_copy_host_to_gpu: hipMemcpy failed: ret={ret}, "
                    f"gpu=0x{int(gpu_ptr):x}, host=0x{int(host_ptr):x}, "
                    f"len={length}"
                )
        logger.info(
            f"_copy_host_to_gpu: copied {len(self._host_staging_ptrs)} "
            f"buffers (total {sum(self._host_staging_lens)} bytes)"
        )
        # Re-register host staging buffers to refresh RDMA NIC mapping.
        # The hipMemcpy DMA may invalidate the NIC's IOMMU mappings.
        try:
            self.engine.batch_deregister(self._host_staging_ptrs)
            self.engine.batch_register(
                self._host_staging_ptrs, self._host_staging_lens
            )
            logger.info("_copy_host_to_gpu: re-registered host staging buffers")
        except Exception as e:
            logger.warning(f"_copy_host_to_gpu: re-registration failed: {e}")

'''

# Insert after deregister_buffer_to_engine method
# Find the end of deregister_buffer_to_engine (next blank line + method/class)
DEREG_PATTERN = r'(    def deregister_buffer_to_engine\(self\):.*?\n\n)'

match = re.search(DEREG_PATTERN, src, re.DOTALL)
if match:
    insert_point = match.end()
    src = src[:insert_point] + COPY_METHOD + src[insert_point:]
    patched = True
    print("patch_host_staging: added _copy_host_to_gpu() method")
else:
    print("patch_host_staging: WARNING - deregister_buffer_to_engine not found")

# --- 4. Patch MooncakeKVReceiver.poll() to call _copy_host_to_gpu() ---
# Current poll() has:
#         if status in (KVPoll.Success, KVPoll.Failed):
#             self.conclude_state = status
#         elif status == KVPoll.WaitingForInput:
#
# We need to add _copy_host_to_gpu() call when status is Success.

OLD_POLL = """        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
        elif status == KVPoll.WaitingForInput:
            timeout_result = self._check_waiting_timeout()
            if timeout_result is not None:
                return timeout_result

        if status == KVPoll.Success:
            # AMD HIP: RDMA writes via GDR/dmabuf are NOT automatically coherent"""

NEW_POLL = """        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            if (
                status == KVPoll.Success
                and hasattr(self.kv_mgr, "_host_staging_buffers")
            ):
                self.kv_mgr._copy_host_to_gpu()
        elif status == KVPoll.WaitingForInput:
            timeout_result = self._check_waiting_timeout()
            if timeout_result is not None:
                return timeout_result

        if status == KVPoll.Success:
            # AMD HIP: RDMA writes via GDR/dmabuf are NOT automatically coherent"""

# This poll() pattern appears in MooncakeKVReceiver (the second poll() in the file)
# We need to be careful to only patch the receiver's poll, not the sender's
# The sender's poll() has a different structure (checks Bootstrapping, not WaitingForInput)

if OLD_POLL in src:
    src = src.replace(OLD_POLL, NEW_POLL, 1)
    patched = True
    print("patch_host_staging: patched MooncakeKVReceiver.poll()")
else:
    # Try without the comment line (might be different in container)
    OLD_POLL_ALT = """        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
        elif status == KVPoll.WaitingForInput:
            timeout_result = self._check_waiting_timeout()
            if timeout_result is not None:
                return timeout_result

        if status == KVPoll.Success:"""

    NEW_POLL_ALT = """        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            if (
                status == KVPoll.Success
                and hasattr(self.kv_mgr, "_host_staging_buffers")
            ):
                self.kv_mgr._copy_host_to_gpu()
        elif status == KVPoll.WaitingForInput:
            timeout_result = self._check_waiting_timeout()
            if timeout_result is not None:
                return timeout_result

        if status == KVPoll.Success:"""

    if OLD_POLL_ALT in src:
        src = src.replace(OLD_POLL_ALT, NEW_POLL_ALT, 1)
        patched = True
        print("patch_host_staging: patched MooncakeKVReceiver.poll() (alt anchor)")
    else:
        print("patch_host_staging: WARNING - poll() anchor not found")

# --- 5. Verify and write ---
if not patched:
    print("patch_host_staging: NO PATCHES APPLIED - check warnings above")
    sys.exit(0)

if src == original_src:
    print("patch_host_staging: source unchanged after patching")
    sys.exit(0)

# Syntax check
try:
    py_compile.compile(CONN_PATH, doraise=True)
except py_compile.PyCompileError:
    pass  # The file on disk is still the old version

# Write to a temp file first, then verify, then move
tmp_path = CONN_PATH + ".patched"
with open(tmp_path, "w") as f:
    f.write(src)

# Verify the patched file compiles
try:
    py_compile.compile(tmp_path, doraise=True)
    os.rename(tmp_path, CONN_PATH)
    print(f"patch_host_staging: SUCCESS - patched {CONN_PATH}")
except py_compile.PyCompileError as e:
    os.unlink(tmp_path)
    print(f"patch_host_staging: FAILED - syntax error in patched file: {e}")
    sys.exit(1)
