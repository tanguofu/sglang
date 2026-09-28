"""Compat shim: upstream aiter 4f613c2 moved fp8_mqa_logits from
aiter/ops/flydsl/kernels/ to aiter/ops/flydsl/kernels/mqa_logits/.
sglang (mi308x-1p1d-src) still imports the old path; re-export until
the sglang side moves.
"""

from aiter.ops.flydsl.kernels.mqa_logits.fp8_mqa_logits import *  # noqa: F401,F403
from aiter.ops.flydsl.kernels.mqa_logits.fp8_mqa_logits import flydsl_fp8_mqa_logits
