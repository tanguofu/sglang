#!/usr/bin/env python3
"""
Parse py-spy raw output and aggregate decode time by operation category.

py-spy raw format (one sample per line):
  Sample <idx> <timestamp_ns> <tid> <thread_id> <frame0>;<frame1>;...;<frameN>
Each frame: "function_name (file:line)" or "function_name:file:line".
Frames are root-first, so the LAST frame is the leaf (currently executing).

Classifies each sample by scanning frames from leaf upward; the first frame
matching a category keyword (in precedence order) assigns the category.
"""
import re
import sys
import collections

# (category, [keywords]) in precedence order. Checked against function name
# AND file path, case-insensitive. First matching category (leaf-first) wins.
CATEGORIES = [
    ("allreduce", ["allreduce", "all_reduce", "allgather", "all_gather",
                   "reducescatter", "reduce_scatter", "nccl", "rccl",
                   "broadcast", "aiter_allreduce", "fused_all_reduce",
                   "_all_reduce", "_allreduce"]),
    ("moe", ["fused_moe", "moe", "expert", "dispatch", "gather", "scatter",
             "topk_gating", "grouped_moe", "moe_align", "moe_fused"]),
    ("attention", ["attention", "flash", "dsa", "mha", "mla", "paged",
                   "radix_attention", "trtllm", "attn", "decode_attention",
                   "qkv", "rope", "rotary"]),
    ("norm", ["rmsnorm", "rms_norm", "add_rms", "fused_qk", "fused_norm",
              "layernorm", "layer_norm", "norm"]),
    ("sampler", ["sample", "sampling", "logits", "argmax", "compute_logits",
                 "top_p", "top_k", "softmax"]),
    ("quant", ["quant", "dequant", "act_quant", "per_token", "scaled_mm",
               "weight_dequant"]),
    ("gemm", ["gemm", "matmul", "linear", "addmm", "bmm", "aiter_gemm",
              "sgemm", "grouped_gemm", "mm"]),
]


def classify(frames):
    """frames: list of (func, file). Leaf is last. Return category."""
    for func, fpath in reversed(frames):
        text = (func + " " + fpath).lower()
        for cat, kws in CATEGORIES:
            for kw in kws:
                if kw in text:
                    return cat, func
    return "other", (frames[-1][0] if frames else "?")


def parse_frame(tok):
    # tok like "function_name (path/file.py:123)" or "function_name:file:line"
    m = re.match(r"^(.*?)\s*(?:\((.*?):(\d+)\)|:(.*):(\d+))?\s*$", tok)
    if m:
        func = m.group(1).strip()
        fpath = (m.group(2) or m.group(4) or "").strip()
        return func, fpath
    return tok, ""


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data/prof19/pyspy_rank0.raw"
    cat_counts = collections.Counter()
    cat_funcs = collections.defaultdict(collections.Counter)
    leaf_funcs = collections.Counter()
    n = 0
    n_samples = 0
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if not line.startswith("Sample"):
                continue
            parts = line.split(" ", 5)
            if len(parts) < 6:
                continue
            frame_str = parts[5]
            frames = [parse_frame(t) for t in frame_str.split(";") if t.strip()]
            if not frames:
                continue
            n_samples += 1
            cat, f = classify(frames)
            cat_counts[cat] += 1
            cat_funcs[cat][f] += 1
            leaf_funcs[frames[-1][0]] += 1
            n += 1

    total = sum(cat_counts.values()) or 1
    print(f"total samples: {n_samples}")
    print()
    print(f"{'category':<12} {'samples':>9} {'% of decode':>12}  top leaf functions")
    print("-" * 100)
    for cat, _ in CATEGORIES + [("other", [])]:
        c = cat_counts.get(cat, 0)
        if c == 0:
            continue
        pct = 100.0 * c / total
        tops = ", ".join(f"{fn}({cnt})" for fn, cnt in cat_funcs[cat].most_common(3))
        print(f"{cat:<12} {c:>9} {pct:>11.1f}%  {tops}")

    print()
    print("=== top 20 leaf functions overall ===")
    for fn, c in leaf_funcs.most_common(20):
        print(f"  {c:>6}  {100.0*c/total:5.1f}%  {fn}")


if __name__ == "__main__":
    main()
