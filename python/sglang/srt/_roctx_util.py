"""Tiny roctx / torch.cuda.nvtx shim used by dsv4-308x indexer probe.

Picks the first available backend in this order:
  1. rocprofiler_sdk.roctx (when rocprofiler-sdk is importable)
  2. torch.cuda.nvtx (works on ROCm via the runtime's NVTX -> roctx bridge,
     and is always available when torch is built with CUDA/HIP)

If neither is available, the functions are silently no-ops. The caller
already wraps each call in try/except so a raised exception from a missing
backend never reaches the request hot path.
"""
from __future__ import annotations


def _resolve_push():
    try:
        from rocprofiler_sdk.roctx import range_push  # type: ignore

        return ("roctx", range_push)
    except Exception:
        pass
    try:
        import torch.cuda.nvtx as _nvtx  # type: ignore

        return ("nvtx", _nvtx.range_push)
    except Exception:
        return ("noop", _noop_push)


def _resolve_pop():
    try:
        from rocprofiler_sdk.roctx import range_pop  # type: ignore

        return ("roctx", range_pop)
    except Exception:
        pass
    try:
        import torch.cuda.nvtx as _nvtx  # type: ignore

        return ("nvtx", _nvtx.range_pop)
    except Exception:
        return ("noop", _noop_pop)


_PUSH = _resolve_push()
_POP = _resolve_pop()


def _noop_push(_msg: str) -> None:
    return None


def _noop_pop() -> None:
    return None


def range_push(msg: str) -> None:
    _, fn = _PUSH
    try:
        fn(msg)
    except Exception:
        pass


def range_pop() -> None:
    _, fn = _POP
    try:
        fn()
    except Exception:
        pass
