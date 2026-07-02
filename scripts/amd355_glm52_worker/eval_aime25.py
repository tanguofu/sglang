#!/usr/bin/env python3
"""AIME 2025 precision evaluation for GLM-5.2 on AMD MI355X.

Reproduces the official SGLang cookbook methodology:
  sgl-eval run aime25 --n-repeats 16 --max-tokens 64000
    --temperature 1.0 --top-p 0.95 --thinking

Target: pass@1 avg-of-16 = 87.7% (SGLang cookbook, H200, FP8+BF16-KV).

Usage (inside container):
  python3 /data/eval_aime25.py --base-url http://localhost:30000/v1 \
    --model /data/models/GLM-5.2-FP8 --n-repeats 16 --max-tokens 64000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI

QUERY_TEMPLATE = (
    "Solve the following AIME (American Invitational Mathematics Examination) "
    "problem step by step. The last line of your response should be of the form "
    "Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.\n\n"
    "Note: AIME answers are always integers from 000 to 999 (inclusive). "
    "If you get a non-integer answer, you likely made a computational error.\n\n"
    "{question}\n\n"
    "Remember to put your answer on its own line after \"Answer:\", and express "
    "your answer as an integer from 000 to 999."
).strip()

ANSWER_PATTERN = r"Answer:\s*(\d+)"

# Matches the last "Answer: XXX" occurrence in the text.
ANSWER_PATTERN_LAST = r"Answer:\s*([0-9]+)"


def normalize_answer(answer: str | None) -> str | None:
    """Normalize AIME answer to a canonical integer string (0-999)."""
    if answer is None:
        return None
    answer = str(answer).strip()
    try:
        num = int(float(answer))
        if 0 <= num <= 999:
            return str(num)
    except (ValueError, TypeError):
        pass
    return answer


def extract_answer(text: str) -> str | None:
    """Extract the last 'Answer: XXX' from the model response."""
    matches = re.findall(ANSWER_PATTERN_LAST, text)
    if matches:
        return matches[-1].strip()
    return None


def load_aime25() -> list[dict]:
    """Load AIME 2025 Part I + II from HuggingFace (opencompass/AIME2025)."""
    ds1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    ds2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
    examples = []
    for ds in (ds1, ds2):
        for row in ds:
            examples.append({"question": row["question"], "answer": str(row["answer"])})
    return examples


def query_once(
    client: OpenAI,
    model: str,
    question: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
) -> str:
    """Send a single chat completion request with thinking mode enabled."""
    prompt = QUERY_TEMPLATE.format(question=question)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        timeout=timeout,
    )
    content = resp.choices[0].message.content or ""
    return content


def eval_problem(
    client: OpenAI,
    model: str,
    example: dict,
    n_repeats: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    idx: int,
    total: int,
) -> dict:
    """Run n_repeats attempts for one problem, return per-problem results."""
    correct_answer = normalize_answer(example["answer"])
    results = []
    n_correct = 0

    for attempt in range(n_repeats):
        try:
            content = query_once(
                client, model, example["question"], max_tokens, temperature, top_p, timeout
            )
            extracted = extract_answer(content)
            normalized = normalize_answer(extracted)
            is_correct = normalized == correct_answer
            if is_correct:
                n_correct += 1
            results.append(
                {
                    "attempt": attempt,
                    "extracted": extracted,
                    "normalized": normalized,
                    "correct": is_correct,
                    "response_len": len(content),
                }
            )
        except Exception as e:
            results.append({"attempt": attempt, "error": str(e), "correct": False})

    pass1 = n_correct / n_repeats
    # pass@n: 1 if any attempt correct
    pass_n = 1.0 if n_correct > 0 else 0.0
    # majority vote
    from collections import Counter

    valid_answers = [
        r["normalized"] for r in results if r.get("normalized") is not None
    ]
    majority = Counter(valid_answers).most_common(1)[0][0] if valid_answers else None
    majority_correct = majority == correct_answer

    print(
        f"[{idx + 1}/{total}] answer={correct_answer} "
        f"pass@1={pass1:.2f} ({n_correct}/{n_repeats}) "
        f"pass@{n_repeats}={pass_n:.0f} majority={'✓' if majority_correct else '✗'}",
        flush=True,
    )

    return {
        "question_idx": idx,
        "correct_answer": correct_answer,
        "pass1": pass1,
        "pass_n": pass_n,
        "majority_correct": majority_correct,
        "n_correct": n_correct,
        "n_repeats": n_repeats,
        "attempts": results,
    }


def main():
    parser = argparse.ArgumentParser(description="AIME 2025 precision eval for GLM-5.2")
    parser.add_argument("--base-url", default="http://localhost:30000/v1")
    parser.add_argument("--model", default="/data/models/GLM-5.2-FP8")
    parser.add_argument("--n-repeats", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num-problems", type=int, default=0, help="0 = all 30")
    parser.add_argument("--out-dir", default="/data/eval_results")
    parser.add_argument("--concurrency", type=int, default=1, help="parallel problems")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    print("Loading AIME 2025 dataset...", flush=True)
    examples = load_aime25()
    if args.num_problems > 0:
        examples = examples[: args.num_problems]
    total = len(examples)
    print(f"Loaded {total} problems. Running {args.n_repeats} repeats each.", flush=True)

    all_results = []
    start = time.time()

    if args.concurrency <= 1:
        for i, ex in enumerate(examples):
            r = eval_problem(
                client, args.model, ex, args.n_repeats, args.max_tokens,
                args.temperature, args.top_p, args.timeout, i, total,
            )
            all_results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    eval_problem, client, args.model, ex, args.n_repeats,
                    args.max_tokens, args.temperature, args.top_p,
                    args.timeout, i, total,
                ): i
                for i, ex in enumerate(examples)
            }
            for f in as_completed(futures):
                all_results.append(f.result())

    elapsed = time.time() - start

    # Aggregate
    avg_pass1 = sum(r["pass1"] for r in all_results) / len(all_results)
    avg_pass_n = sum(r["pass_n"] for r in all_results) / len(all_results)
    majority_acc = sum(r["majority_correct"] for r in all_results) / len(all_results)

    print("\n" + "=" * 60)
    print(f"AIME 2025 Results ({total} problems, {args.n_repeats} repeats each)")
    print(f"  pass@1 avg-of-{args.n_repeats}: {avg_pass1:.1%}  (target: 87.7%)")
    print(f"  pass@{args.n_repeats}:              {avg_pass_n:.1%}  (target: 100%)")
    print(f"  majority@{args.n_repeats}:          {majority_acc:.1%}  (target: 93.3%)")
    print(f"  Elapsed: {elapsed / 60:.1f} min")
    print("=" * 60)

    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"aime25_results_{ts}.json"
    summary = {
        "benchmark": "AIME 2025",
        "n_problems": total,
        "n_repeats": args.n_repeats,
        "pass1_avg": avg_pass1,
        "pass_n_avg": avg_pass_n,
        "majority_acc": majority_acc,
        "elapsed_sec": elapsed,
        "config": {
            "model": args.model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        "per_problem": all_results,
    }
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results saved to {out_file}", flush=True)

    # Exit code: 0 if pass1 within 3% of target
    if avg_pass1 >= 0.847:
        print("✅ PASS: pass@1 within tolerance of 87.7% target")
        sys.exit(0)
    else:
        print("❌ FAIL: pass@1 below 84.7% (3% tolerance from 87.7% target)")
        sys.exit(1)


if __name__ == "__main__":
    main()
