#!/usr/bin/env python3
"""HIP fusion + dual stream patch for SGLang 0702 image (v6.2).
v6.2: fixes cos_sin_cache 4D->2D squeeze for AITER rotary embedding."""

INDEXER = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"
GLM4 = "/sgl-workspace/sglang/python/sglang/srt/models/glm4_moe.py"

def patch_file(path, patches):
    with open(path, "r") as f:
        content = f.read()
    for desc, old, new in patches:
        if old in content:
            content = content.replace(old, new, 1)
            print(f"[OK] Patched: {desc}")
        elif new in content:
            print(f"[SKIP] Already patched: {desc}")
        else:
            print(f"[WARN] Pattern not found: {desc}")
    with open(path, "w") as f:
        f.write(content)

TE8 = "        if isinstance(x, tuple) and len(x) == 3:\n            x = x[2]\n"
TE12 = "            if isinstance(x, tuple) and len(x) == 3:\n                x = x[2]\n"

indexer_patches = [
    ("dsa_indexer fusion on HIP",
     "_use_dsa_indexer_fusion = _is_cuda and not envs.SGLANG_DISABLE_DSA_INDEXER_FUSION.get()",
     "_use_dsa_indexer_fusion = not envs.SGLANG_DISABLE_DSA_INDEXER_FUSION.get()"),
    ("dual stream threshold on HIP",
     "DUAL_STREAM_TOKEN_THRESHOLD = 1024 if _is_cuda else 0",
     "DUAL_STREAM_TOKEN_THRESHOLD = 1024"),
    ("dsv4+dsv32 imports on HIP (inner guard)",
     "if _is_cuda or _is_hip:\n    if _is_cuda:\n        from sglang.jit_kernel.dsv4 import fused_q_indexer_rope_first_quant\n        from sglang.jit_kernel.dsv32 import (\n            fused_k_indexer_norm_rope,\n            fused_k_indexer_norm_rope_store,\n        )",
     "if _is_cuda or _is_hip:\n    if True:\n        from sglang.jit_kernel.dsv4 import fused_q_indexer_rope_first_quant\n        from sglang.jit_kernel.dsv32 import (\n            fused_k_indexer_norm_rope,\n            fused_k_indexer_norm_rope_store,\n        )"),
    ("_indexer_cos_sin_cache property -> cached val + k_norm f32",
     "    @property\n    def _indexer_cos_sin_cache(self) -> torch.Tensor:\n        return self.rotary_emb.cos_sin_cache",
     "    @property\n    def _indexer_cos_sin_cache(self) -> torch.Tensor:\n        return self._cos_sin_cache_val\n\n    @property\n    def _k_norm_weight_f32(self):\n        w = self.k_norm.weight\n        return w if w.dtype == torch.float32 else w.to(torch.float32)\n\n    @property\n    def _k_norm_bias_f32(self):\n        b = self.k_norm.bias\n        return b if b is not None and b.dtype == torch.float32 else (b.to(torch.float32) if b is not None else None)"),
    ("store cos_sin_cache_val in __init__ (aiter compatible, 4D->2D squeeze)",
     "        self.block_size = block_size\n        self.scale_fmt = scale_fmt",
     "        if hasattr(self.rotary_emb, 'cos_sin_cache'):\n            self._cos_sin_cache_val = self.rotary_emb.cos_sin_cache\n        else:\n            self._cos_sin_cache_val = torch.cat([\n                self.rotary_emb.cos_cache, self.rotary_emb.sin_cache\n            ], dim=-1).reshape(self.rotary_emb.cos_cache.shape[0], -1).to(torch.float32)\n        self.block_size = block_size\n        self.scale_fmt = scale_fmt"),
    ("_fused_k_weights tuple extraction",
     "    def _fused_k_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:\n        kw, _ = self.wk_weights_proj(x)",
     "    def _fused_k_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:\n" + TE8 + "        kw, _ = self.wk_weights_proj(x)"),
    ("_fused_q_prepare non-dual-stream tuple extraction",
     "        if self.alt_stream is None or not enable_dual_stream:\n            kw, _ = self.wk_weights_proj(x)",
     "        if self.alt_stream is None or not enable_dual_stream:\n" + TE12 + "            kw, _ = self.wk_weights_proj(x)"),
    ("_fused_q_prepare dual-stream tuple extraction",
     "        kw, _ = self.wk_weights_proj(x)\n        key, weights_raw = kw.split([self.head_dim, self.n_heads], dim=-1)\n        if num_tokens is not None:\n            key = key[:num_tokens]\n            weights_raw = weights_raw[:num_tokens]\n\n        current_stream.wait_stream(self.alt_stream)",
     "        " + TE8.strip() + "\n            kw, _ = self.wk_weights_proj(x)\n        key, weights_raw = kw.split([self.head_dim, self.n_heads], dim=-1)\n        if num_tokens is not None:\n            key = key[:num_tokens]\n            weights_raw = weights_raw[:num_tokens]\n\n        current_stream.wait_stream(self.alt_stream)"),
    ("use float32 weight/bias in fused_k_indexer_norm_rope",
     "        key = fused_k_indexer_norm_rope(\n            key_raw,\n            self.k_norm.weight,\n            self.k_norm.bias,\n            self.k_norm.variance_epsilon,\n            self._indexer_cos_sin_cache,\n            positions,\n        )",
     "        key = fused_k_indexer_norm_rope(\n            key_raw,\n            self._k_norm_weight_f32,\n            self._k_norm_bias_f32,\n            self.k_norm.variance_epsilon,\n            self._indexer_cos_sin_cache,\n            positions,\n        )"),
    ("use float32 weight/bias in fused_k_indexer_norm_rope_store",
     "            fused_k_indexer_norm_rope_store(\n                key_raw,\n                pool.get_index_k_with_scale_buffer(layer_id=layer_id),\n                out_cache_loc,\n                self.k_norm.weight,\n                self.k_norm.bias,\n                self.k_norm.variance_epsilon,\n                self._indexer_cos_sin_cache,\n                positions,\n                page_size,\n            )",
     "            fused_k_indexer_norm_rope_store(\n                key_raw,\n                pool.get_index_k_with_scale_buffer(layer_id=layer_id),\n                out_cache_loc,\n                self._k_norm_weight_f32,\n                self._k_norm_bias_f32,\n                self.k_norm.variance_epsilon,\n                self._indexer_cos_sin_cache,\n                positions,\n                page_size,\n            )"),
    ("fix rotary_emb.cos_sin_cache[positions] -> _indexer_cos_sin_cache",
     "                cos_sin = self.rotary_emb.cos_sin_cache[positions]",
     "                cos_sin = self._indexer_cos_sin_cache[positions]"),
    ("fix rotary_emb.cos_sin_cache.index_select -> _indexer_cos_sin_cache",
     "                    self.rotary_emb.cos_sin_cache.index_select(0, positions)",
     "                    self._indexer_cos_sin_cache.index_select(0, positions)"),
]

patch_file(INDEXER, indexer_patches)

with open(GLM4, "r") as f:
    glm4_content = f.read()
if "self.alt_stream = torch.cuda.Stream()" in glm4_content and "if _is_cuda" not in glm4_content.split("self.alt_stream = torch.cuda.Stream()")[0][-50:]:
    print("[SKIP] Already patched: GLM4 MoE alt_stream on HIP")
else:
    patch_file(GLM4, [
        ("GLM4 MoE alt_stream on HIP",
         "self.alt_stream = torch.cuda.Stream() if _is_cuda else None",
         "self.alt_stream = torch.cuda.Stream()"),
    ])

print("\n=== HIP fusion + dual stream patch complete (0702 v6.2) ===")
