#!/usr/bin/env python3
"""Patch mori conn.py: align PP KV mem-desc slicing with common/conn.py."""

import os
import sys

MORI_CONN = "/sgl-workspace/sglang/python/sglang/srt/disaggregation/mori/conn.py"


def patch() -> None:
    if not os.path.exists(MORI_CONN):
        print(f"[ERROR] File not found: {MORI_CONN}")
        sys.exit(1)

    with open(MORI_CONN, "r") as f:
        content = f.read()

    marker = "if num_local_layers == len(dst_mem_descs):"
    if marker in content:
        print("[PATCH] Already patched - mori PP KV slice fix already applied")
        return

    old_mla = """        src_descs = self.kv_mem_descs
        num_local_layers = len(src_descs)
        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + num_local_layers
        if end_layer > len(dst_mem_descs):
            raise ValueError(
                "Destination MLA KV descriptors do not match prefill pp configuration"
            )
        dst_slice = dst_mem_descs[start_layer:end_layer]
        return src_descs, dst_slice, num_local_layers"""

    new_mla = """        src_descs = self.kv_mem_descs
        num_local_layers = len(src_descs)
        if num_local_layers == len(dst_mem_descs):
            return src_descs, dst_mem_descs, num_local_layers

        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + num_local_layers
        if end_layer > len(dst_mem_descs):
            raise ValueError(
                "Destination MLA KV descriptors do not match prefill pp configuration"
            )
        dst_slice = dst_mem_descs[start_layer:end_layer]
        return src_descs, dst_slice, num_local_layers"""

    if old_mla not in content:
        print("[ERROR] Could not find MLA mem-desc slice block to patch")
        sys.exit(1)
    content = content.replace(old_mla, new_mla)
    print("[PATCH] Patched _get_mla_mem_desc_slices PP fast path")

    old_mha = """        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + num_local_layers
        dst_total_layers = len(dst_mem_descs) // 2
        if len(dst_mem_descs) < 2 or end_layer > dst_total_layers:
            raise ValueError(
                "Destination KV descriptors do not match prefill pp configuration"
            )
        dst_k_descs = dst_mem_descs[start_layer:end_layer]
        dst_v_descs = dst_mem_descs[
            dst_total_layers + start_layer : dst_total_layers + end_layer
        ]
        return src_k_descs, src_v_descs, dst_k_descs, dst_v_descs, num_local_layers"""

    new_mha = """        dst_total_layers = len(dst_mem_descs) // 2
        if len(dst_mem_descs) < 2:
            raise ValueError(
                "Destination KV descriptors do not match prefill pp configuration"
            )
        if num_local_layers == dst_total_layers:
            dst_k_descs = dst_mem_descs[:dst_total_layers]
            dst_v_descs = dst_mem_descs[dst_total_layers:]
        else:
            start_layer = self.kv_args.prefill_start_layer
            end_layer = start_layer + num_local_layers
            if end_layer > dst_total_layers:
                raise ValueError(
                    "Destination KV descriptors do not match prefill pp configuration"
                )
            dst_k_descs = dst_mem_descs[start_layer:end_layer]
            dst_v_descs = dst_mem_descs[
                dst_total_layers + start_layer : dst_total_layers + end_layer
            ]
        return src_k_descs, src_v_descs, dst_k_descs, dst_v_descs, num_local_layers"""

    if old_mha not in content:
        print("[ERROR] Could not find MHA mem-desc slice block to patch")
        sys.exit(1)
    content = content.replace(old_mha, new_mha)
    print("[PATCH] Patched _get_mha_mem_desc_slices PP fast path")

    with open(MORI_CONN, "w") as f:
        f.write(content)

    print("[PATCH] Successfully patched mori PP KV mem-desc slicing")


if __name__ == "__main__":
    patch()
