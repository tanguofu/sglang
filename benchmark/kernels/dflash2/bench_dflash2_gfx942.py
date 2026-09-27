import argparse
import json
import time
from typing import Any, Dict, List

import torch


def _bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def _aiter_available() -> bool:
    try:
        import aiter  # noqa: F401

        return True
    except ImportError:
        return False


def _bench_gemm(
    m: int,
    n: int,
    k: int,
    warmup: int,
    iters: int,
    use_aiter: bool,
) -> Dict[str, Any]:
    hidden = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    torch_result = torch.matmul(hidden, weight.T).float()

    result: Dict[str, Any] = {
        "op": "gemm",
        "m": m,
        "n": n,
        "k": k,
        "torch_ms": _bench(
            lambda: torch.matmul(hidden, weight.T), warmup, iters
        ),
    }

    if use_aiter:
        from aiter.tuned_gemm import gemm_a16w16

        aiter_result = gemm_a16w16(hidden, weight, otype=torch.float32).to(
            torch.bfloat16
        ).float()
        result["aiter_ms"] = _bench(
            lambda: gemm_a16w16(hidden, weight, otype=torch.float32).to(
                torch.bfloat16
            ),
            warmup,
            iters,
        )
        result["max_abs_diff"] = float(
            (aiter_result - torch_result).abs().max().item()
        )
        result["allclose"] = bool(
            torch.allclose(aiter_result, torch_result, atol=2e-2, rtol=2e-2)
        )
        torch_values, torch_indices = torch.topk(torch_result, 16, dim=-1)
        aiter_values, aiter_indices = torch.topk(aiter_result, 16, dim=-1)
        result["topk_index_mismatches"] = int(
            (torch_indices != aiter_indices).sum().item()
        )
        result["topk_max_abs_diff"] = float(
            (torch_values - aiter_values).abs().max().item()
        )

    return result


def _bench_topk(
    m: int,
    n: int,
    k: int,
    warmup: int,
    iters: int,
    use_aiter: bool,
) -> Dict[str, Any]:
    logits = torch.randn(m, n, device="cuda", dtype=torch.float32)
    torch_values, torch_indices = torch.topk(logits, k, dim=-1)

    result: Dict[str, Any] = {
        "op": "topk",
        "m": m,
        "n": n,
        "k": k,
        "torch_ms": _bench(lambda: torch.topk(logits, k, dim=-1), warmup, iters),
    }

    if use_aiter:
        from aiter.ops.topk import top_k_per_row_decode

        seq_lens = torch.full((m,), n, device="cuda", dtype=torch.int32)
        indices = torch.empty((m, k), device="cuda", dtype=torch.int32)

        def run():
            top_k_per_row_decode(
                logits,
                1,
                seq_lens,
                indices,
                m,
                logits.stride(0),
                logits.stride(1),
                k=k,
                stable=True,
            )
            values = logits.gather(1, indices.long())
            values, order = torch.sort(values, descending=True, stable=True)
            sorted_indices = indices.gather(1, order)
            indices.copy_(sorted_indices)
            return values

        values = run()
        result["aiter_ms"] = _bench(run, warmup, iters)
        result["index_mismatches"] = int(
            (indices.long() != torch_indices).sum().item()
        )
        result["value_mismatches"] = int(
            (values != torch_values).sum().item()
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms", type=int, nargs="+", default=[7, 14, 21, 28, 35, 42, 49, 56, 64])
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--vocab-size", type=int, default=154880)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if args.tp_size <= 0:
        raise ValueError("--tp-size must be positive")
    if args.vocab_size % args.tp_size:
        raise ValueError("--vocab-size must be divisible by --tp-size")

    use_aiter = _aiter_available()
    shard_vocab_size = args.vocab_size // args.tp_size
    results: List[Dict[str, Any]] = []

    for m in args.ms:
        results.append(
            _bench_gemm(
                m,
                shard_vocab_size,
                args.hidden_size,
                args.warmup,
                args.iters,
                use_aiter,
            )
        )
        results.append(
            _bench_topk(
                m,
                shard_vocab_size,
                args.top_k,
                args.warmup,
                args.iters,
                use_aiter,
            )
        )

    print(json.dumps({"aiter_available": use_aiter, "results": results}, indent=2))


if __name__ == "__main__":
    main()
