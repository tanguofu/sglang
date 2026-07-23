#!/usr/bin/env python3
"""Aggregate py-spy raw folded-stack samples into a self-time breakdown.

Reads prof32-pyspy-raw.txt (folded format: `frame1;...;frameN count`),
computes per-leaf self-time, groups leaves into decode phases, and prints
the top functions and phase shares.
"""
import re
import sys
from collections import defaultdict

RAW = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/guofutan/ti-cloud/teamai/ti-cloud-teamai/deployments/glm52-tp8-0718/results/prof32/prof32-pyspy-raw.txt"

# leaf frame looks like: func_name (path/file.py:line)
FRAME_RE = re.compile(r"^(.*?) \((.*?):(\d+)\)$")


def leaf_name(frame):
    m = FRAME_RE.match(frame)
    if not m:
        return frame, ""
    return m.group(1), m.group(2)


# Phase classification by keywords in the full stack (checked leaf-first, but
# we classify the leaf frame's function/file).
def classify(func, fpath):
    s = (func + " " + fpath).lower()
    if any(k in s for k in [
        "eagle_worker", "eagle", "speculat", "verify", "target_verify",
        "draft_model",
    ]):
        return "EAGLE-verify"
    if any(k in s for k in [
        "moe", "fused_moe", "moe_sorting", "moe_runner", "topk", "gate",
        "routing", "expert",
    ]):
        return "MoE-routing"
    if any(k in s for k in [
        "dsa_backend", "radix_attention", "attention", "tilelang",
        "flashattn", "flash_attn", "attn_backend", "triton_attn",
    ]):
        return "attention-launch"
    if any(k in s for k in [
        "sampler", "sampling", "sample", "logits_processor", "process_logits",
    ]):
        return "sampling"
    if any(k in s for k in [
        "scheduler", "schedule_batch", "run_batch", "event_loop",
        "dispatch_event", "prepare_dp_attn_batch", "get_next_batch",
        "schedule", "prefill_schedule", "decode_schedule",
    ]):
        return "scheduler"
    if any(k in s for k in [
        "ipc", "zmq", "socket", "comm", "recv", "send", "tokenizer",
        "detokenizer", "io_struct",
    ]):
        return "IPC/tokenizer"
    if any(k in s for k in [
        "allreduce", "all_reduce", "distributed", "barrier", "coalesced",
    ]):
        return "allreduce/comm"
    if any(k in s for k in [
        "torch/cuda", "current_stream", "current_device", "_lazy_init",
        "is_initialized", "_get_device", "torch._utils", "torch._ops",
        "torch_guard", "nvtx_utils", "contextlib",
    ]):
        return "torch-overhead"
    if any(k in s for k in [
        "deepseek_v2", "model_runner", "eager_runner", "forward",
        "rmsnorm", "rotary", "rope", "norm", "linear", "quantization",
        "fp8", "w8a8", "block",
    ]):
        return "model-forward"
    return "other"


self_time = defaultdict(int)
phase_time = defaultdict(int)
total = 0

with open(RAW) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        # folded: "frame1;frame2;...;frameN count"
        try:
            stack, cnt = line.rsplit(" ", 1)
            cnt = int(cnt)
        except ValueError:
            continue
        frames = stack.split(";")
        if not frames:
            continue
        leaf = frames[-1]
        func, fpath = leaf_name(leaf)
        self_time[(func, fpath)] += cnt
        phase = classify(func, fpath)
        phase_time[phase] += cnt
        total += cnt

print(f"Total samples: {total}  (each ~1 sample = 1/50 s = 20ms)\n")

print("=== Phase breakdown (by leaf self-time) ===")
for phase, cnt in sorted(phase_time.items(), key=lambda x: -x[1]):
    pct = 100.0 * cnt / total if total else 0
    print(f"  {phase:20s} {cnt:6d}  {pct:5.1f}%")

print("\n=== Top 25 functions by self-time ===")
print(f"  {'self%':>6s}  {'samples':>7s}  function (file)")
items = sorted(self_time.items(), key=lambda x: -x[1])
for (func, fpath), cnt in items[:25]:
    pct = 100.0 * cnt / total if total else 0
    short = fpath.split("/")[-1] if fpath else "?"
    print(f"  {pct:6.2f}  {cnt:7d}  {func} ({short})")
