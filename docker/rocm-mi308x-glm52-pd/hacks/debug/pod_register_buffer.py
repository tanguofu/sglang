    def register_buffer_to_engine(self):
        if os.environ.get("SGLANG_PD_HOST_STAGING") == "1":
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
                logger.info(
                    f"Host staging: registered {len(self._host_staging_ptrs)} "
                    f"host buffers for KV data (total "
                    f"{sum(self._host_staging_lens)} bytes), "
                    f"replaced kv_data_ptrs with host addresses"
                )
        elif self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
            self.engine.batch_register(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            )

        # Batch register auxiliary data buffers
