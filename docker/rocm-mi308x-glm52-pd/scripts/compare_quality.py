#!/usr/bin/env python3
"""Compare deterministic outputs from two OpenAI-compatible chat endpoints."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


PROMPTS = [
    "用一句话说明 7+8 等于多少。",
    "把 'The model is fast.' 翻译成中文。",
    "写一个 Python 函数返回两个数的最大值。",
    "解释什么是 KV cache，不超过三句话。",
    "列出 HTTP 200、404、500 的含义。",
    "计算 (17-5)*3 的值。",
    "把 'prefill' 和 'decode' 翻译成中文。",
    "用一句话解释 speculative decoding。",
    "写一个 bash 命令列出当前目录的 JSON 文件。",
    "解释 TCP 和 UDP 的核心区别，不超过三句话。",
    "写一个 SQL 查询统计 users 表的行数。",
    "把 '12345' 转成整数并加 1。",
    "用一句话说明为什么温度设为 0。",
    "写一个正则表达式匹配 11 位手机号。",
    "解释什么是 load balancing，不超过两句话。",
    "给出 3 的前五个倍数。",
    "用一句话解释什么是 tokenizer。",
    "写一个 Python list comprehension 生成 1 到 10 的平方。",
    "把 'hello world' 改成大写。",
    "用一句话说明什么是端到端延迟。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-url", required=True)
    parser.add_argument("--right-url", required=True)
    parser.add_argument("--left-model", required=True)
    parser.add_argument("--right-model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def request_one(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        choice = payload["choices"][0]
        message = choice["message"]
        return {
            "ok": True,
            "status": response.status,
            "content": message.get("content") or "",
            "reasoning": message.get("reasoning_content") or "",
            "token_ids": choice.get("token_ids"),
            "finish_reason": choice.get("finish_reason"),
            "elapsed_s": round(time.time() - started, 3),
        }
    except urllib.error.HTTPError as error:
        return {
            "ok": False,
            "status": error.code,
            "error": error.read(1000).decode("utf-8", "replace"),
            "elapsed_s": round(time.time() - started, 3),
        }
    except Exception as error:
        return {
            "ok": False,
            "status": 0,
            "error": f"{type(error).__name__}: {error}",
            "elapsed_s": round(time.time() - started, 3),
        }


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        count += 1
    return count


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    left_key = os.environ.get("LEFT_API_KEY")
    right_key = os.environ.get("RIGHT_API_KEY")
    records = []
    for index, prompt in enumerate(PROMPTS):
        left = request_one(
            args.left_url,
            args.left_model,
            prompt,
            args.max_tokens,
            args.timeout,
            left_key,
        )
        right = request_one(
            args.right_url,
            args.right_model,
            prompt,
            args.max_tokens,
            args.timeout,
            right_key,
        )
        left_ids = left.get("token_ids") or tokenizer.encode(
            left.get("content") or "", add_special_tokens=False
        )
        right_ids = right.get("token_ids") or tokenizer.encode(
            right.get("content") or "", add_special_tokens=False
        )
        records.append(
            {
                "index": index,
                "prompt": prompt,
                "left": left,
                "right": right,
                "text_exact": left.get("content") == right.get("content"),
                "token_exact": left_ids == right_ids,
                "first_token_equal": bool(left_ids and right_ids and left_ids[0] == right_ids[0]),
                "common_prefix_tokens": common_prefix_length(left_ids, right_ids),
                "left_tokens": len(left_ids),
                "right_tokens": len(right_ids),
            }
        )
        print(
            json.dumps(
                {
                    "index": index,
                    "ok": left["ok"] and right["ok"],
                    "text_exact": records[-1]["text_exact"],
                    "token_exact": records[-1]["token_exact"],
                    "common_prefix_tokens": records[-1]["common_prefix_tokens"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "prompt_count": len(records),
        "request_ok_count": sum(r["left"]["ok"] and r["right"]["ok"] for r in records),
        "text_exact_count": sum(r["text_exact"] for r in records),
        "token_exact_count": sum(r["token_exact"] for r in records),
        "first_token_equal_count": sum(r["first_token_equal"] for r in records),
        "text_exact_rate": sum(r["text_exact"] for r in records) / len(records),
        "token_exact_rate": sum(r["token_exact"] for r in records) / len(records),
        "first_token_equal_rate": sum(r["first_token_equal"] for r in records) / len(records),
        "records": records,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}))


if __name__ == "__main__":
    main()
