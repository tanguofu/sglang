#!/usr/bin/env python3
"""Port patch for aiter_backend.py: accept **kwargs in forward_extend/forward_decode.

Problem: GLM-5.2 is a DSA model (GlmMoeDsaForCausalLM with index_topk set). The
MLA forward path (forward_mla.py) unconditionally passes `topk_indices` via
`attn_mqa(..., topk_indices=...)` whenever the indexer produced it. The kwarg
flows through RadixAttention.forward -> base_attn_backend.forward ->
backend.forward_decode(**kwargs). AiterAttnBackend.forward_decode() only declares
`(self, q, k, v, layer, forward_batch, save_kv_cache, sinks=None)` and has no
**kwargs, so passing topk_indices raises:
    TypeError: AiterAttnBackend.forward_decode() got an unexpected keyword
    argument 'topk_indices'

This crashes EAGLE speculative decode on the decode pod (the EAGLE draft layers
run attn_mqa with topk_indices). Prefill is also affected because the same
forward_extend path is used.

Fix: add **kwargs to both forward_extend and forward_decode signatures. The aiter
backend is not the DSA backend (it does not consume topk_indices for sparse
attention), so silently accepting and ignoring the kwarg is correct and matches
the base class contract (base_attn_backend.AttentionBackend.forward_decode has
**kwargs). Other kwargs that may flow through (q_rope, k_rope, cos_sin_cache,
is_neox, llama_4_scaling) are also ignored by the aiter backend's MLA-unaware
fast path, which is the intended behavior for non-DSA backends.

Idempotent: detects the `_dsa_kwargs_compat` marker and skips if already applied.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/port-post1/aiter_backend.py")
src = path.read_text()

GUARD_MARKER = "_dsa_kwargs_compat"

if GUARD_MARKER in src:
    print(f"[aiter_backend] already patched ({GUARD_MARKER} present), skipped")
    sys.exit(0)

# Patch 1: forward_extend signature
# Old signature ends with `sinks=None,\n    ):` — we add **kwargs before the colon.
extend_old = (
    "    def forward_extend(\n"
    "        self,\n"
    "        q: torch.Tensor,\n"
    "        k: torch.Tensor,\n"
    "        v: torch.Tensor,\n"
    "        layer: RadixAttention,\n"
    "        forward_batch: ForwardBatch,\n"
    "        save_kv_cache=True,\n"
    "        sinks=None,\n"
    "    ):"
)
extend_new = (
    "    def forward_extend(  # " + GUARD_MARKER + "\n"
    "        self,\n"
    "        q: torch.Tensor,\n"
    "        k: torch.Tensor,\n"
    "        v: torch.Tensor,\n"
    "        layer: RadixAttention,\n"
    "        forward_batch: ForwardBatch,\n"
    "        save_kv_cache=True,\n"
    "        sinks=None,\n"
    "        **kwargs,\n"
    "    ):"
)

# Patch 2: forward_decode signature
decode_old = (
    "    def forward_decode(\n"
    "        self,\n"
    "        q: torch.Tensor,\n"
    "        k: torch.Tensor,\n"
    "        v: torch.Tensor,\n"
    "        layer: RadixAttention,\n"
    "        forward_batch: ForwardBatch,\n"
    "        save_kv_cache=True,\n"
    "        sinks=None,\n"
    "    ):"
)
decode_new = (
    "    def forward_decode(  # " + GUARD_MARKER + "\n"
    "        self,\n"
    "        q: torch.Tensor,\n"
    "        k: torch.Tensor,\n"
    "        v: torch.Tensor,\n"
    "        layer: RadixAttention,\n"
    "        forward_batch: ForwardBatch,\n"
    "        save_kv_cache=True,\n"
    "        sinks=None,\n"
    "        **kwargs,\n"
    "    ):"
)

changed = []

if extend_old in src:
    src = src.replace(extend_old, extend_new, 1)
    changed.append("forward_extend: +**kwargs")
else:
    changed.append("forward_extend: target not found (already patched or source changed)")

if decode_old in src:
    src = src.replace(decode_old, decode_new, 1)
    changed.append("forward_decode: +**kwargs")
else:
    changed.append("forward_decode: target not found (already patched or source changed)")

path.write_text(src)
print(f"[aiter_backend] patched: " + ", ".join(changed))
