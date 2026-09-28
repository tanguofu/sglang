"""Cheap GDR coherence helpers for HIP L2 / PCIe posted writes.

Prefill: buffer_wbl2 so NIC GDR reads fresh KV, then 8B RDMA READ flush.
Decode: buffer_inv so kernels refetch HBM after GDR WRITE.

Safe to import from scheduler workers. HSACO is baked next to this module
(or overridden via SGLANG_PD_GDR_FLUSH_HSACO).
"""
from __future__ import annotations

import ctypes
import logging
import os
from typing import Optional

logger = logging.getLogger("gdr_l2_flush")

_PKG_HSACO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdr_l2_flush.hsaco")
_HOST_HSACO = "/data/mooncake-patched/gdr_l2_flush.hsaco"


def _default_hsaco() -> str:
    if os.path.exists(_PKG_HSACO):
        return _PKG_HSACO
    return _HOST_HSACO


HSACO = os.environ.get("SGLANG_PD_GDR_FLUSH_HSACO", _default_hsaco())

_hip = None
_mod = None
_fn_wb = None
_fn_inv = None
_sink_addr = None
_sink_registered = False


def _hip_lib():
    global _hip
    if _hip is None:
        _hip = ctypes.CDLL("libamdhip64.so")
        _hip.hipSetDevice.argtypes = [ctypes.c_int]
        _hip.hipSetDevice.restype = ctypes.c_int
        _hip.hipDeviceSynchronize.restype = ctypes.c_int
        _hip.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        _hip.hipModuleLoad.restype = ctypes.c_int
        _hip.hipModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        _hip.hipModuleGetFunction.restype = ctypes.c_int
        _hip.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        _hip.hipModuleLaunchKernel.restype = ctypes.c_int
        _hip.hipMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        _hip.hipMallocHost.restype = ctypes.c_int
    return _hip


def _load(gpu_id: int) -> bool:
    global _mod, _fn_wb, _fn_inv
    if _fn_wb is not None and _fn_inv is not None:
        return True
    if not os.path.exists(HSACO):
        logger.error("gdr_l2_flush: hsaco missing: %s", HSACO)
        return False
    hip = _hip_lib()
    hip.hipSetDevice(ctypes.c_int(int(gpu_id)))
    mod = ctypes.c_void_p()
    rc = hip.hipModuleLoad(ctypes.byref(mod), HSACO.encode())
    if rc != 0:
        logger.error("gdr_l2_flush: hipModuleLoad rc=%s path=%s", rc, HSACO)
        return False
    fn_wb = ctypes.c_void_p()
    fn_inv = ctypes.c_void_p()
    rc_wb = hip.hipModuleGetFunction(ctypes.byref(fn_wb), mod, b"gdr_l2_wb")
    rc_inv = hip.hipModuleGetFunction(ctypes.byref(fn_inv), mod, b"gdr_l2_inv")
    if rc_wb != 0 or rc_inv != 0:
        logger.error("gdr_l2_flush: getFunction wb=%s inv=%s", rc_wb, rc_inv)
        return False
    _mod, _fn_wb, _fn_inv = mod, fn_wb, fn_inv
    return True


def _launch(fn, gpu_id: int) -> bool:
    hip = _hip_lib()
    hip.hipSetDevice(ctypes.c_int(int(gpu_id)))
    rc = hip.hipModuleLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, None, None, None)
    if rc != 0:
        logger.error("gdr_l2_flush: launch rc=%s", rc)
        return False
    rc = hip.hipDeviceSynchronize()
    if rc != 0:
        logger.error("gdr_l2_flush: sync rc=%s", rc)
        return False
    return True


def writeback(gpu_id: int = 0) -> bool:
    if not _load(gpu_id):
        return False
    return _launch(_fn_wb, gpu_id)


def invalidate(gpu_id: int = 0) -> bool:
    if not _load(gpu_id):
        return False
    return _launch(_fn_inv, gpu_id)


def rdma_read_flush(engine, session_id: str, sink: int, src: int, length: int = 8) -> int:
    """8B RDMA READ via the inner C++ TransferEngine.

    SGLang's MooncakeTransferEngine wrapper only exposes transfer_sync /
    batch_transfer_sync (WRITE). Calling transfer_sync_read on the wrapper
    raises AttributeError on every PD transfer.
    """
    inner = getattr(engine, "engine", None)
    fn = None
    if inner is not None and hasattr(inner, "transfer_sync_read"):
        fn = inner.transfer_sync_read
    elif hasattr(engine, "transfer_sync_read"):
        fn = engine.transfer_sync_read
    if fn is None:
        raise AttributeError(
            f"no transfer_sync_read on wrapper={type(engine).__name__} "
            f"inner={type(inner).__name__ if inner is not None else None}"
        )
    return int(fn(session_id, int(sink), int(src), int(length)))


def ensure_read_sink(engine, gpu_id: int = 0) -> Optional[int]:
    """64B host sink for the 8-byte loopback RDMA READ."""
    global _sink_addr, _sink_registered
    if _sink_addr is not None:
        return _sink_addr
    hip = _hip_lib()
    hip.hipSetDevice(ctypes.c_int(int(gpu_id)))
    ptr = ctypes.c_void_p()
    rc = hip.hipMallocHost(ctypes.byref(ptr), ctypes.c_size_t(64))
    if rc != 0 or not ptr.value:
        logger.error("gdr_l2_flush: hipMallocHost sink rc=%s", rc)
        return None
    registered = False
    reg = None
    for attempt in ("batch_register", "register", "inner_register_memory"):
        try:
            if attempt == "batch_register" and hasattr(engine, "batch_register"):
                reg = engine.batch_register([int(ptr.value)], [64])
            elif attempt == "register" and hasattr(engine, "register"):
                reg = engine.register(int(ptr.value), 64)
            elif attempt == "inner_register_memory" and hasattr(
                getattr(engine, "engine", None), "register_memory"
            ):
                reg = engine.engine.register_memory(int(ptr.value), 64)
            else:
                continue
            registered = True
            break
        except Exception as e:
            logger.warning("gdr_l2_flush: sink %s failed: %s", attempt, e)
    if not registered:
        logger.error("gdr_l2_flush: no working register API on %s", type(engine).__name__)
        return None
    if reg not in (0, None):
        logger.warning("gdr_l2_flush: sink register ret=%s (continuing)", reg)
    _sink_addr = int(ptr.value)
    _sink_registered = True
    logger.info("gdr_l2_flush: read-sink registered at 0x%x", _sink_addr)
    return _sink_addr
