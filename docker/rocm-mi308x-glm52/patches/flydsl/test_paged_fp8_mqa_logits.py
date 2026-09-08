import argparse

import torch

from .fp8_mqa_logits import flydsl_fp8_paged_mqa_logits


def _make_inputs(batch_size, next_n, max_seq_len, device, page_size=1, preshuffle=False):
    rows = batch_size * next_n
    num_heads = 64
    head_dim = 128
    num_blocks = (max_seq_len + page_size - 1) // page_size
    num_tokens = num_blocks * page_size

    torch.manual_seed(1234)
    q = torch.randint(
        0, 128, (rows, num_heads, head_dim), dtype=torch.uint8, device=device
    ).view(torch.float8_e4m3fnuz)
    weights = torch.rand(rows, num_heads, device=device)

    base_context_lens = torch.randint(
        max_seq_len // 2, max_seq_len + 1, (batch_size,), device=device
    ).to(torch.int32)
    context_lens = base_context_lens.repeat_interleave(next_n, dim=0)

    base_block_tables = torch.stack(
        [torch.randperm(num_blocks, device=device) for _ in range(batch_size)]
    ).to(torch.int32)
    block_tables = base_block_tables.repeat_interleave(next_n, dim=0)

    if page_size == 1:
        kv = torch.randint(
            0, 128, (num_tokens, head_dim + 4), dtype=torch.uint8, device=device
        )
        scales = (0.98 + 0.04 * torch.rand(num_tokens, device=device)).float()
        kv[:, head_dim:] = scales.view(torch.uint8).reshape(num_tokens, 4)
        kv = kv.view(torch.float8_e4m3fnuz)
        kv_flat = kv.reshape(-1)
        logical_k = kv.view(-1, head_dim + 4)[:, :head_dim]
        return q, kv_flat, scales, weights, context_lens, block_tables, logical_k

    assert preshuffle
    from aiter.ops.cache import (
        cp_gather_indexer_k_quant_cache,
        indexer_k_quant_and_cache,
    )

    k = torch.randn((num_tokens, head_dim), dtype=torch.bfloat16, device=device)
    kv = torch.zeros(
        (num_blocks, page_size, head_dim + 4),
        dtype=torch.float8_e4m3fnuz,
        device=device,
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    indexer_k_quant_and_cache(
        k,
        kv,
        slot_mapping,
        128,
        "ue8m0",
        preshuffle=True,
    )

    linear_block_table = torch.arange(
        num_blocks, dtype=torch.int32, device=device
    ).view(1, -1)
    cu_seq_lens = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    logical_k = torch.empty(
        (num_tokens, head_dim), dtype=torch.float8_e4m3fnuz, device=device
    )
    scales = torch.empty((num_tokens, 1), dtype=torch.float32, device=device)
    cp_gather_indexer_k_quant_cache(
        kv,
        logical_k,
        scales,
        linear_block_table,
        cu_seq_lens,
        preshuffle=True,
    )
    kv_flat = kv.reshape(-1)
    kv_scales = kv.view(torch.float32).reshape(num_blocks, page_size * 33)[
        :, page_size * 32 :
    ]
    return q, kv_flat, kv_scales, weights, context_lens, block_tables, logical_k


def _reference(q, logical_k, scales, weights, context_lens, block_tables, page_size):
    if page_size == 1:
        k = logical_k[block_tables].to(torch.float32)
        page_scales = scales[block_tables]
    else:
        num_blocks = logical_k.shape[0] // page_size
        k = logical_k.view(num_blocks, page_size, -1)[block_tables].reshape(
            block_tables.shape[0], -1, logical_k.shape[-1]
        ).to(torch.float32)
        page_scales = scales.view(num_blocks, page_size)[block_tables].reshape(
            block_tables.shape[0], -1
        )
    scores = torch.einsum("rhd,rnd->rhn", q.to(torch.float32), k)
    scores = torch.relu(scores * page_scales[:, None, :])
    scores = scores * weights[:, :, None]
    return scores.sum(dim=1)


def _valid_error(output, reference, context_lens):
    valid = torch.arange(output.shape[1], device=output.device)[None, :] < context_lens[:, None]
    diff = (output - reference).abs()[valid]
    return diff.max().item(), diff.mean().item()


def _benchmark(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def _run_case(args, page_size, preshuffle):
    device = torch.device("cuda")
    q, kv, scales, weights, context_lens, block_tables, logical_k = _make_inputs(
        args.batch_size, args.next_n, args.max_seq_len, device, page_size, preshuffle
    )
    reference = _reference(
        q, logical_k, scales, weights, context_lens, block_tables, page_size
    )

    variant = "mfma_r4_w4" if args.next_n == 4 else "mfma_r1_w4"
    output = flydsl_fp8_paged_mqa_logits(
        q,
        kv,
        scales,
        weights,
        context_lens,
        block_tables,
        args.max_seq_len,
        variant=variant,
        page_size=page_size,
        preshuffle=preshuffle,
    )
    max_error, mean_error = _valid_error(output, reference, context_lens)
    print(
        f"flydsl correctness page_size={page_size}: "
        f"max_abs={max_error:.6f} mean_abs={mean_error:.6f} "
        f"variant={variant} rows={output.shape[0]} cols={output.shape[1]}"
    )
    valid = torch.arange(output.shape[1], device=output.device)[None, :] < context_lens[:, None]
    reference_scale = reference[valid].abs().max().item()
    tolerance = max(0.05, 2e-6 * reference_scale)
    if max_error > tolerance:
        raise RuntimeError(f"FlyDSL paged MQA error too high: {max_error}")

    fly_ms = _benchmark(
        lambda: flydsl_fp8_paged_mqa_logits(
            q,
            kv,
            scales,
            weights,
            context_lens,
            block_tables,
            args.max_seq_len,
            variant=variant,
            page_size=page_size,
            preshuffle=preshuffle,
        ),
        args.iters,
    )
    print(f"flydsl time page_size={page_size}: {fly_ms:.3f} ms")

    from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

    num_blocks = scales.shape[0]
    kv_cache = kv.view(num_blocks, page_size, 1, q.shape[2] + 4)
    q_aiter = q.unsqueeze(1)
    aiter_logits = torch.empty(
        (q.shape[0], args.max_seq_len), dtype=torch.float32, device=q.device
    )

    def run_aiter():
        deepgemm_fp8_paged_mqa_logits(
            q_aiter,
            kv_cache,
            weights,
            aiter_logits,
            context_lens,
            block_tables,
            args.max_seq_len,
            Preshuffle=preshuffle,
            KVBlockSize=page_size,
        )

    run_aiter()
    aiter_max_error, aiter_mean_error = _valid_error(
        aiter_logits, reference, context_lens
    )
    print(
        f"aiter correctness page_size={page_size}: "
        f"max_abs={aiter_max_error:.6f} mean_abs={aiter_mean_error:.6f}"
    )
    aiter_ms = _benchmark(run_aiter, args.iters)
    print(
        f"aiter time page_size={page_size}: {aiter_ms:.3f} ms "
        f"speedup={aiter_ms / fly_ms:.3f}x"
    )

    static_context_lens = context_lens.clone()
    static_block_tables = block_tables.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = flydsl_fp8_paged_mqa_logits(
            q,
            kv,
            scales,
            weights,
            static_context_lens,
            static_block_tables,
            args.max_seq_len,
            variant=variant,
            page_size=page_size,
            preshuffle=preshuffle,
        )
    torch.cuda.synchronize()

    new_context_lens = torch.clamp(
        context_lens + torch.randint_like(context_lens, -128, 129), 1, args.max_seq_len
    )
    static_context_lens.copy_(new_context_lens)
    graph.replay()
    torch.cuda.synchronize()
    new_reference = _reference(
        q, logical_k, scales, weights, new_context_lens, static_block_tables, page_size
    )
    graph_max_error, _ = _valid_error(graph_output, new_reference, new_context_lens)
    print(
        f"cuda graph replay correctness page_size={page_size}: "
        f"max_abs={graph_max_error:.6f}"
    )
    graph_reference_scale = new_reference[
        torch.arange(new_reference.shape[1], device=new_reference.device)[None, :]
        < new_context_lens[:, None]
    ].abs().max().item()
    if graph_max_error > max(0.05, 2e-6 * graph_reference_scale):
        raise RuntimeError(f"CUDA graph replay error too high: {graph_max_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--next-n", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    _run_case(args, page_size=1, preshuffle=False)
    _run_case(args, page_size=64, preshuffle=True)


if __name__ == "__main__":
    main()
