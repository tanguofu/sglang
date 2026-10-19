#!/usr/bin/env python3
"""200K TTFT/TFOT benchmark for the 1P1D PD router."""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import time
import urllib.error
import urllib.request
from typing import Any


FILLER = (
    "人工智能是计算机科学的一个分支，它致力于研究、开发用于模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。"
    "人工智能的研究范畴广泛，包括机器学习、深度学习、自然语言处理、计算机视觉、知识表示、自动推理、机器人学等多个领域。"
    "近年来，随着大数据技术的发展和计算能力的提升，基于深度学习的方法在图像识别、语音识别、自然语言理解等任务上取得了突破性进展。"
)

WORDS = (
    "system", "model", "token", "cache", "batch", "kernel", "router",
    "prefill", "decode", "throughput", "latency", "benchmark", "stream",
    "request", "response", "server", "client", "network", "memory", "compute",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30002")
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--tokenizer", default="/data/model/glm53-fp8")
    parser.add_argument("--target-tokens", type=int, default=200000)
    parser.add_argument("--cold-runs", type=int, default=3)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--target-worker", default="0")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--random-filler", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--output", default="/tmp/bench_200k_ttft_tfot.json")
    return parser.parse_args()


def load_api_key() -> str:
    if os.environ.get("SGLANG_BENCH_API_KEY"):
        return os.environ["SGLANG_BENCH_API_KEY"]
    with open("/proc/1/cmdline", "rb") as handle:
        args = handle.read().split(b"\0")
    for index, arg in enumerate(args):
        if arg == b"--api-key" and index + 1 < len(args):
            return args[index + 1].decode()
    raise RuntimeError("API key not found; set SGLANG_BENCH_API_KEY")


def trim_to_tokens(tokenizer: Any, text: str, target_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= target_tokens:
        return text
    return tokenizer.decode(ids[:target_tokens])


def build_messages(
    tokenizer: Any,
    target_tokens: int,
    salt: str,
    random_filler: bool,
) -> list[dict[str, str]]:
    task = "\n\nDo not call tools. Think briefly, then answer with one short sentence."
    prefix = f"[{salt}] "
    overhead = len(tokenizer.encode(prefix + task, add_special_tokens=False))
    filler_target = max(0, target_tokens - overhead)
    if random_filler:
        filler = " ".join(
            random.choice(WORDS)
            for _ in range(filler_target * 2 + 1024)
        )
    else:
        unit = f"{salt} {FILLER}"
        unit_tokens = len(tokenizer.encode(unit, add_special_tokens=False))
        repeats = filler_target // unit_tokens + 2
        filler = trim_to_tokens(tokenizer, unit * repeats, filler_target)
    if random_filler:
        filler = trim_to_tokens(tokenizer, filler, filler_target)
    user_content = prefix + filler + task
    return [
        {"role": "system", "content": "You are a precise assistant."},
        {"role": "user", "content": user_content},
    ]


def stream_chat(
    url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    target_worker: str,
    ignore_eos: bool,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": ignore_eos,
        "chat_template_kwargs": {"enable_thinking": True},
        "thinking": {"type": "enabled"},
    }
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    if target_worker:
        request.add_header("X-SMG-Target-Worker", target_worker)
    started = time.time()
    ttft = None
    tfot = None
    first_kind = None
    content = ""
    reasoning = ""
    usage = {}
    finish = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            buffer = b""
            while True:
                piece = response.read(4096)
                if not piece:
                    break
                buffer += piece
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    text = line.decode("utf-8", "replace").strip()
                    if not text.startswith("data:"):
                        continue
                    payload = text[5:].strip()
                    if payload == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    reasoning_piece = delta.get("reasoning_content") or ""
                    content_piece = delta.get("content") or ""
                    if ttft is None and (reasoning_piece or content_piece):
                        ttft = time.time() - started
                        first_kind = "reasoning_content" if reasoning_piece else "content"
                    if tfot is None and content_piece:
                        tfot = time.time() - started
                    content += content_piece
                    reasoning += reasoning_piece
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    if chunk.get("usage"):
                        usage = chunk["usage"]
    except urllib.error.HTTPError as error:
        return {
            "ok": False,
            "status": error.code,
            "error": error.read(1000).decode("utf-8", "replace"),
            "wall_s": round(time.time() - started, 3),
        }
    except Exception as error:
        return {
            "ok": False,
            "status": 0,
            "error": f"{type(error).__name__}: {error}",
            "wall_s": round(time.time() - started, 3),
        }
    details = usage.get("prompt_tokens_details") or {}
    return {
        "ok": True,
        "status": 200,
        "wall_s": round(time.time() - started, 3),
        "ttft_s": None if ttft is None else round(ttft, 3),
        "tfot_s": None if tfot is None else round(tfot, 3),
        "first_kind": first_kind,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "cached_tokens": details.get("cached_tokens") if isinstance(details, dict) else None,
        "finish": finish,
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    api_key = load_api_key()
    results = []
    for cold_index in range(args.cold_runs):
        salt = "".join(random.choices(string.ascii_lowercase, k=10))
        messages = build_messages(
            tokenizer,
            args.target_tokens,
            salt,
            args.random_filler,
        )
        for run_kind in ["cold", *["warm"] * args.warm_runs]:
            record = {
                "pair": cold_index + 1,
                "kind": run_kind,
                "salt": salt,
                "target_worker": args.target_worker,
            }
            record.update(
                stream_chat(
                    args.url,
                    api_key,
                    args.model,
                    messages,
                    args.max_tokens,
                    args.timeout,
                    args.target_worker,
                    args.ignore_eos,
                )
            )
            results.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"output": args.output, "count": len(results)}), flush=True)


if __name__ == "__main__":
    main()
