import re

filepath = "python/sglang/srt/disaggregation/mooncake/conn.py"
with open(filepath, "r") as f:
    content = f.read()

# === Patch 1: register_buffer_to_engine — add GPU→host mapping ===
old_register = '''                self.kv_args.kv_data_ptrs = list(self._host_staging_ptrs)
                logger.info(
                    f"Host staging: registered {len(self._host_staging_ptrs)} "
                    f"host buffers for KV data (total "
                    f"{sum(self._host_staging_lens)} bytes), "
                    f"replaced kv_data_ptrs with host addresses"
                )'''

new_register = '''                self.kv_args.kv_data_ptrs = list(self._host_staging_ptrs)
                # Build GPU→host address mapping for O(1) lookup in _transfer_data.
                # Each entry maps a GPU buffer base to its (host_base, length) so
                # that any GPU address within [gpu_base, gpu_base+length) can be
                # translated to the corresponding host staging address.
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
                )'''

assert old_register in content, "register_buffer_to_engine pattern not found"
content = content.replace(old_register, new_register)

# === Patch 2: _transfer_data — use pre-registered staging buffers ===
old_transfer = '''        if os.environ.get("SGLANG_PD_HOST_STAGING") == "1":
            import ctypes

            hip_lib = ctypes.CDLL("libamdhip64.so")
            host_ptrs_to_free = []
            need_register = []
            reg_lens = []
            final_src_addrs = []
            for src_addr, dst_addr, length in transfer_blocks:
                # Allocate pinned host memory for RDMA access
                host_ptr = ctypes.c_void_p()
                alloc_ret = hip_lib.hipMallocHost(
                    ctypes.byref(host_ptr), ctypes.c_size_t(length)
                )
                if alloc_ret != 0:
                    # hipMallocHost failed — use original address (CPU data)
                    final_src_addrs.append(int(src_addr))
                    continue
                ret = hip_lib.hipMemcpy(
                    host_ptr,
                    ctypes.c_void_p(int(src_addr)),
                    ctypes.c_size_t(length),
                    ctypes.c_int(2),  # hipMemcpyDeviceToHost
                )
                if ret != 0:
                    # Not a GPU address — free host buffer, use original
                    hip_lib.hipFreeHost(host_ptr)
                    final_src_addrs.append(int(src_addr))
                else:
                    host_ptrs_to_free.append(host_ptr)
                    need_register.append(host_ptr.value)
                    reg_lens.append(length)
                    final_src_addrs.append(host_ptr.value)
            if need_register:
                self.engine.batch_register(need_register, reg_lens)
            ret = self.engine.batch_transfer_sync(
                mooncake_session_id, final_src_addrs, list(dst_addrs), list(lengths)
            )
            if need_register:
                self.engine.batch_deregister(need_register)
            for ptr in host_ptrs_to_free:
                hip_lib.hipFreeHost(ptr)
            return ret
            return ret

        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )'''

new_transfer = '''        if os.environ.get("SGLANG_PD_HOST_STAGING") == "1" and hasattr(
            self, "_gpu_to_host_map"
        ):
            import ctypes

            hip_lib = ctypes.CDLL("libamdhip64.so")
            final_src_addrs = []
            for src_addr, dst_addr, length in transfer_blocks:
                src_addr_int = int(src_addr)
                # Look up the pre-registered host staging buffer that
                # corresponds to this GPU address.  The GPU address sits
                # at an offset within one of the kv_data pools registered
                # in register_buffer_to_engine; the host staging buffer
                # has the same layout, so we compute the matching host
                # offset.
                host_src = None
                for gpu_base, (host_base, buf_len) in self._gpu_to_host_map.items():
                    if gpu_base <= src_addr_int < gpu_base + buf_len:
                        host_src = host_base + (src_addr_int - gpu_base)
                        break
                if host_src is not None:
                    # Copy GPU data into the pre-registered host staging
                    # buffer at the matching offset, then transfer from
                    # there.  No per-transfer register/deregister — the
                    # buffer is already registered in the segment.
                    ret = hip_lib.hipMemcpy(
                        ctypes.c_void_p(host_src),
                        ctypes.c_void_p(src_addr_int),
                        ctypes.c_size_t(length),
                        ctypes.c_int(2),  # hipMemcpyDeviceToHost
                    )
                    if ret != 0:
                        logger.warning(
                            f"hipMemcpy D2H failed for src={hex(src_addr_int)} "
                            f"host={hex(host_src)} len={length} ret={ret}, "
                            f"using GPU address directly"
                        )
                        final_src_addrs.append(src_addr_int)
                    else:
                        final_src_addrs.append(host_src)
                else:
                    final_src_addrs.append(src_addr_int)
            return self.engine.batch_transfer_sync(
                mooncake_session_id, final_src_addrs, list(dst_addrs), list(lengths)
            )

        if os.environ.get("SGLANG_PD_HOST_STAGING") == "1":
            # Fallback: no pre-registered staging map (e.g. custom mem pool).
            # Use the original per-transfer alloc/register/deregister path.
            import ctypes

            hip_lib = ctypes.CDLL("libamdhip64.so")
            host_ptrs_to_free = []
            need_register = []
            reg_lens = []
            final_src_addrs = []
            for src_addr, dst_addr, length in transfer_blocks:
                host_ptr = ctypes.c_void_p()
                alloc_ret = hip_lib.hipMallocHost(
                    ctypes.byref(host_ptr), ctypes.c_size_t(length)
                )
                if alloc_ret != 0:
                    final_src_addrs.append(int(src_addr))
                    continue
                ret = hip_lib.hipMemcpy(
                    host_ptr,
                    ctypes.c_void_p(int(src_addr)),
                    ctypes.c_size_t(length),
                    ctypes.c_int(2),
                )
                if ret != 0:
                    hip_lib.hipFreeHost(host_ptr)
                    final_src_addrs.append(int(src_addr))
                else:
                    host_ptrs_to_free.append(host_ptr)
                    need_register.append(host_ptr.value)
                    reg_lens.append(length)
                    final_src_addrs.append(host_ptr.value)
            if need_register:
                self.engine.batch_register(need_register, reg_lens)
            ret = self.engine.batch_transfer_sync(
                mooncake_session_id, final_src_addrs, list(dst_addrs), list(lengths)
            )
            if need_register:
                self.engine.batch_deregister(need_register)
            for ptr in host_ptrs_to_free:
                hip_lib.hipFreeHost(ptr)
            return ret

        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )'''

assert old_transfer in content, "_transfer_data pattern not found"
content = content.replace(old_transfer, new_transfer)

with open(filepath, "w") as f:
    f.write(content)

print("Patch applied successfully")
