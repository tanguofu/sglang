#!/usr/bin/env python3
"""Patch mooncake/conn.py for selective host staging copy.

Replaces the full 66.7 GB _copy_host_to_gpu with a selective version that
only copies the KV pages actually transferred (determined by kv_indices).

Also modifies send_metadata to store kv_indices and poll() to pass them
to _copy_host_to_gpu.

Idempotent: detects existing patches and updates them.
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

with open(CONN_PATH, "r") as f:
    src = f.read()

original_src = src
patched = False

# --- 1. Replace _copy_host_to_gpu with selective version ---
# Match v1 (no args) or a previous v2 signature.
OLD_COPY = re.search(
    r'    def _copy_host_to_gpu\(self(?:, kv_indices=None)?\):.*?(?=\n    def |\n\n    [a-z]|\nclass )',
    src,
    re.DOTALL,
)

NEW_COPY = '''    def _copy_host_to_gpu(self, kv_indices=None):
        """Copy transferred KV pages from host staging to GPU.

        Never copies the full pool: a tens-of-GB hipMemcpy DMA can
        invalidate bnxt_re IOMMU mappings and kill RDMA QPs.
        Missing kv_indices skips the copy rather than falling back.
        """
        import ctypes

        hip_lib = ctypes.CDLL("libamdhip64.so")
        gpu_id = getattr(self.kv_args, "gpu_id", 0)
        hip_lib.hipSetDevice(ctypes.c_int(gpu_id))

        if kv_indices is None or len(kv_indices) == 0:
            logger.warning(
                "_copy_host_to_gpu: skip copy — no kv_indices "
                "(refusing full-pool hipMemcpy that kills bnxt_re QPs)"
            )
            return

        page_size = self.kv_args.page_size
        kv_item_lens = self.kv_args.kv_item_lens

        indices = sorted(set(int(i) for i in kv_indices))
        groups = []
        start = indices[0]
        end = indices[0] + 1
        for idx in indices[1:]:
            if idx == end:
                end = idx + 1
            else:
                groups.append((start, end - start))
                start = idx
                end = idx + 1
        groups.append((start, end - start))

        total_copied = 0
        for buf_idx, (host_ptr, gpu_ptr) in enumerate(
            zip(self._host_staging_ptrs, self._gpu_ptrs)
        ):
            if buf_idx >= len(kv_item_lens):
                continue
            item_len = kv_item_lens[buf_idx]
            if item_len == 0:
                continue
            for start_idx, count in groups:
                offset = start_idx * page_size * item_len
                length = count * page_size * item_len
                ret = hip_lib.hipMemcpy(
                    ctypes.c_void_p(int(gpu_ptr) + offset),
                    ctypes.c_void_p(int(host_ptr) + offset),
                    ctypes.c_size_t(length),
                    ctypes.c_int(1),  # hipMemcpyHostToDevice
                )
                if ret != 0:
                    logger.error(
                        f"_copy_host_to_gpu: hipMemcpy failed ret={ret} "
                        f"buf={buf_idx} offset={offset} len={length} dev={gpu_id}"
                    )
                total_copied += length
        logger.info(
            f"_copy_host_to_gpu: selective copy {total_copied} bytes "
            f"for {len(indices)} pages ({len(groups)} groups) dev={gpu_id}"
        )

'''

if OLD_COPY:
    src = src[:OLD_COPY.start()] + NEW_COPY + src[OLD_COPY.end():]
    patched = True
    print("patch_host_staging_v2: replaced _copy_host_to_gpu with selective version")
else:
    print("patch_host_staging_v2: WARNING - _copy_host_to_gpu not found")
    sys.exit(1)

# --- 2. Modify send_metadata to store kv_indices ---
# Find: self.init_time = time.time()  (at end of send_metadata)
# Add: self._dst_kv_indices = kv_indices  before it

OLD_SEND_META_END = """        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            if (
                status == KVPoll.Success
                and hasattr(self.kv_mgr, "_host_staging_buffers")
            ):
                self.kv_mgr._copy_host_to_gpu()"""

NEW_SEND_META_END = """        self._dst_kv_indices = kv_indices
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            if (
                status == KVPoll.Success
                and hasattr(self.kv_mgr, "_host_staging_buffers")
            ):
                self.kv_mgr._copy_host_to_gpu(getattr(self, "_dst_kv_indices", None))"""

if OLD_SEND_META_END in src:
    src = src.replace(OLD_SEND_META_END, NEW_SEND_META_END, 1)
    patched = True
    print("patch_host_staging_v2: stored kv_indices in send_metadata + pass to poll")
else:
    # Try alternate: maybe already partially patched
    if "_dst_kv_indices" in src:
        print("patch_host_staging_v2: kv_indices storage already present")
    else:
        print("patch_host_staging_v2: WARNING - send_metadata/poll anchor not found")

def _run_gdr_flush():
    gdr = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patch_gdr_flush.py")
    if os.path.exists(gdr):
        os.system(f"python3 {gdr} {CONN_PATH}")


# --- 3. Verify and write ---
if not patched or src == original_src:
    print("patch_host_staging_v2: NO CHANGES")
    _run_gdr_flush()
    sys.exit(0)

# Syntax check on temp file
tmp_path = CONN_PATH + ".v2patched"
with open(tmp_path, "w") as f:
    f.write(src)

try:
    py_compile.compile(tmp_path, doraise=True)
    os.rename(tmp_path, CONN_PATH)
    print(f"patch_host_staging_v2: SUCCESS - patched {CONN_PATH}")
    _run_gdr_flush()
except py_compile.PyCompileError as e:
    os.unlink(tmp_path)
    print(f"patch_host_staging_v2: FAILED - syntax error: {e}")
    sys.exit(1)
