#!/usr/bin/env python3
"""PoC suite for the isolated PoC-mode worker (:31000 standalone).

Tests (against a single PoC server, no router/PD path):
  1. ping        — short chat completion returns non-empty content
  2. needle      — ~196K-token blob with an embedded code; answer must contain it
  3. decode tps  — streaming generation on the cached 196K context; tps measured
                   from first token to last token (prefill excluded)
  4. accept_len  — speculative accept length from /v1/loads

Usage:
  python3 bench_poc_suite.py --url http://21.151.225.172:31000 --label baseline
Results are written to poc-results-<label>.json (one file per label).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

FILLER = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen "
    "jugs. How vexingly quick daft zebras jump. Sphinx of black quartz, judge "
    "my vow. The five boxing wizards jump quickly. "
)


def chat(url: str, messages, max_tokens: int, timeout: float, stream: bool = False):
    body = {
        "model": "glm-5.3",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def build_needle_prompt(target_tokens: int, code: str) -> list:
    filler = FILLER * (target_tokens // 24 + 8)
    text = (
        f"{filler}\n\n[MEMO] The secret access code for the build system is {code}. "
        "Keep it confidential.\n\n"
        "Repeat the secret access code from the memo above. Answer with the code only."
    )
    return [{"role": "user", "content": text}]


def get_json(url: str, path: str, timeout: float = 30):
    return json.loads(
        urllib.request.urlopen(url + path, timeout=timeout).read()
    )


def run_suite(url: str, label: str, target_tokens: int, gen_tokens: int) -> dict:
    results = {"label": label, "url": url, "ts": time.time()}

    # 1) ping
    t0 = time.time()
    resp = chat(
        url,
        [{"role": "user", "content": "1+1=? Answer with the number only."}],
        max_tokens=256,
        timeout=120,
    )
    d = json.loads(resp.read())
    content = d["choices"][0]["message"].get("content") or ""
    results["ping"] = {
        "ok": bool(content.strip()),
        "content": content.strip()[:40],
        "latency_s": round(time.time() - t0, 2),
    }
    print(f"[ping] ok={results['ping']['ok']} content={results['ping']['content']!r}")

    # 2) needle (cold prefill ~196K)
    code = f"ZK-{label.upper()}-{int(time.time()) % 100000:05d}"
    messages = build_needle_prompt(target_tokens, code)
    t0 = time.time()
    resp = chat(url, messages, max_tokens=512, timeout=1800)
    d = json.loads(resp.read())
    dt = time.time() - t0
    content = d["choices"][0]["message"].get("content") or ""
    usage = d.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    results["needle"] = {
        "ok": code in content,
        "code": code,
        "answer": content.strip()[:60],
        "prompt_tokens": prompt_tokens,
        "total_s": round(dt, 1),
    }
    print(
        f"[needle] ok={results['needle']['ok']} prompt={prompt_tokens} "
        f"total={dt:.1f}s answer={results['needle']['answer']!r}"
    )

    # 3) decode tps on the cached context (streaming, TTFT excludes prefill)
    t0 = time.time()
    ttft = None
    resp = chat(
        url,
        messages,
        max_tokens=gen_tokens,
        timeout=1800,
        stream=True,
    )
    comp = 0
    for line in resp:
        if not line.startswith(b"data: "):
            continue
        payload = line[6:].strip()
        if payload == b"[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
            if ttft is None:
                ttft = time.time() - t0
            comp += 1
    total = time.time() - t0
    decode_s = total - (ttft or 0)
    results["decode"] = {
        "gen_tokens": gen_tokens,
        "ttft_s": round(ttft or 0, 1),
        "decode_s": round(decode_s, 1),
        "tps": round(gen_tokens / decode_s, 2) if decode_s > 0 else 0,
    }
    print(
        f"[decode] ttft={results['decode']['ttft_s']}s "
        f"decode={results['decode']['decode_s']}s tps={results['decode']['tps']}"
    )

    # 4) accept_len
    loads = get_json(url, "/v1/loads")
    spec = loads.get("loads", [{}])[0].get("speculative", {})
    results["accept_len"] = spec.get("accept_length")
    results["accept_rate"] = spec.get("accept_rate")
    print(f"[accept] len={results['accept_len']} rate={results['accept_rate']}")

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="PoC server base URL")
    ap.add_argument("--label", required=True, help="baseline | parity | w1 | w2 ...")
    ap.add_argument("--target-tokens", type=int, default=196000)
    ap.add_argument("--gen-tokens", type=int, default=512)
    args = ap.parse_args()

    results = run_suite(args.url, args.label, args.target_tokens, args.gen_tokens)
    out = f"poc-results-{args.label}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] results -> {out}")


if __name__ == "__main__":
    main()
