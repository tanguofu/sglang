#!/usr/bin/env python3
"""200K Codex/agent-style perf on /v1/chat/completions (cc-switch path).

Never hits /v1/responses.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

ROUTER = "http://sglang-1p1d-router.kube-system:30001"
MODEL = "glm-5.2"
TOK_PATH = "/data/model/glm52-fp8"
OUT = "/data/bench_codex_200k.json"
SALT = str(int(time.time()))

FILLER = (
    "人工智能是计算机科学的一个分支，它致力于研究、开发用于模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。"
    "人工智能的研究范畴广泛，包括机器学习、深度学习、自然语言处理、计算机视觉、知识表示、自动推理、机器人学等多个领域。"
    "近年来，随着大数据技术的发展和计算能力的提升，基于深度学习的方法在图像识别、语音识别、自然语言理解等任务上取得了突破性进展。"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


def _req(body: dict, timeout: float, stream: bool = False):
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(
        ROUTER + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raw = e.read()[:1500]
        try:
            payload = json.loads(raw)
        except Exception:
            payload = raw.decode("utf-8", "replace")
        return e.code, payload, time.time() - t0, None
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}", time.time() - t0, None

    if not stream:
        raw = resp.read()
        dt = time.time() - t0
        try:
            return resp.status, json.loads(raw), dt, None
        except json.JSONDecodeError:
            return resp.status, raw[:500].decode("utf-8", "replace"), dt, None

    ttft = None
    chunks = []
    usage = {}
    finish = ""
    content = ""
    reasoning = ""
    tool_calls = None
    buf = b""
    while True:
        piece = resp.read(4096)
        if not piece:
            break
        buf += piece
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            s = line.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            payload = s[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            chunks.append(obj)
            if ttft is None:
                ttft = time.time() - t0
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
            if delta.get("tool_calls"):
                tool_calls = delta.get("tool_calls")
            if ch.get("finish_reason"):
                finish = ch.get("finish_reason")
            if obj.get("usage"):
                usage = obj["usage"]
    dt = time.time() - t0
    return (
        200,
        {
            "content": content,
            "reasoning": reasoning,
            "finish": finish,
            "usage": usage,
            "tool_calls": tool_calls,
            "n_chunks": len(chunks),
        },
        dt,
        ttft,
    )


def chat(messages, *, max_tokens: int, timeout: float, stream: bool, tools=None):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": True},
        "thinking": {"type": "enabled"},
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    return _req(body, timeout, stream=stream)


def usage_of(payload):
    if not isinstance(payload, dict):
        return {}
    return payload.get("usage") or {}


def rec_of(name, status, payload, dt, ttft, prompt_tokens_est=None):
    usage = usage_of(payload) if not (isinstance(payload, dict) and "usage" in payload) else payload.get("usage") or {}
    if isinstance(payload, dict) and payload.get("choices"):
        msg = (payload.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        finish = (payload.get("choices") or [{}])[0].get("finish_reason")
        tools = msg.get("tool_calls")
    elif isinstance(payload, dict):
        content = payload.get("content") or ""
        reasoning = payload.get("reasoning") or ""
        finish = payload.get("finish")
        tools = payload.get("tool_calls")
        usage = payload.get("usage") or usage
    else:
        content, reasoning, finish, tools = "", "", "", None
    prompt = usage.get("prompt_tokens") or prompt_tokens_est or 0
    comp = usage.get("completion_tokens") or 0
    rtok = usage.get("reasoning_tokens") or 0
    cached = None
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
    in_tps = prompt / dt if dt else 0
    # decode tps: completion tokens after first token
    decode_s = (dt - ttft) if (ttft and dt > ttft) else dt
    out_tps = comp / decode_s if decode_s and comp else 0
    rec = {
        "name": name,
        "status": status,
        "sec": round(dt, 2),
        "ttft_s": None if ttft is None else round(ttft, 2),
        "in_tps": round(in_tps, 1),
        "out_tps": round(out_tps, 1),
        "prompt_tokens": prompt,
        "completion_tokens": comp,
        "reasoning_tokens": rtok,
        "cached_tokens": cached,
        "finish": finish,
        "has_reasoning": bool(reasoning),
        "has_tool_call": bool(tools),
        "content_head": (content or "")[:240],
        "reason_head": (reasoning or "")[:240],
        "payload_err": None if isinstance(payload, dict) else str(payload)[:300],
    }
    print(
        f"{name} HTTP {status} wall={dt:.1f}s ttft={rec['ttft_s']} "
        f"in={rec['in_tps']} tok/s out={rec['out_tps']} tok/s "
        f"prompt={prompt} comp={comp} rtok={rtok} cached={cached} "
        f"finish={finish} tool={bool(tools)}",
        flush=True,
    )
    return rec


def build_repo_blob(tok, target: int) -> str:
    unit = f"[{SALT}] // src/module.py mock repo dump\n" + FILLER + "\n"
    ids = tok.encode(unit, add_special_tokens=False)
    n = max(1, target // max(len(ids), 1))
    blob = unit * n
    # trim by tokens
    ids = tok.encode(blob, add_special_tokens=False)
    if len(ids) > target:
        blob = tok.decode(ids[:target])
    return blob


def main():
    from transformers import AutoTokenizer

    print(f"SALT={SALT} 200K Codex/agent chat path", flush=True)
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    results = {"salt": SALT, "path": "/v1/chat/completions", "turns": []}

    # 0. ping
    st, payload, dt, ttft = chat(
        [{"role": "user", "content": "Reply with the single word pong."}],
        max_tokens=32,
        timeout=60,
        stream=True,
    )
    results["ping"] = rec_of("ping", st, payload, dt, ttft)
    if st != 200:
        json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
        print("PING FAIL", flush=True)
        return

    blob = build_repo_blob(tok, 196000)
    ntok = len(tok.encode(blob, add_special_tokens=False))
    print(f"built repo blob tokens={ntok}", flush=True)
    sys_msg = (
        "You are a coding agent in a large repository. The user pasted a 200K-token "
        "workspace dump. Think, then act with tools when needed."
    )
    task1 = (
        "The dump is tagged "
        + SALT
        + ". Find the tag, then call read_file with path src/main.py. "
        "After the tool call, one sentence on the next edit."
    )
    messages_t1 = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": blob + "\n\n" + task1},
    ]

    # 1. cold 200k agent turn, thinking on, tools, stream for TTFT
    st, payload, dt, ttft = chat(
        messages_t1,
        max_tokens=1024,
        timeout=900,
        stream=True,
        tools=TOOLS,
    )
    results["turns"].append(rec_of("t1_cold_200k_think_tools", st, payload, dt, ttft, ntok))

    # 2. warm follow-up: same dump + short instruction (prefix reuse)
    task2 = "Same workspace. Call shell with command 'git status --short'. One sentence only after thinking."
    messages_t2 = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": blob + "\n\n" + task2},
    ]
    st, payload, dt, ttft = chat(
        messages_t2,
        max_tokens=1024,
        timeout=900,
        stream=True,
        tools=TOOLS,
    )
    results["turns"].append(rec_of("t2_warm_200k_followup", st, payload, dt, ttft, ntok))

    # 3. decode-heavy at 200k KV: force longer completion
    task3 = (
        "Do not call tools. After a short think, list 8 concrete refactors for this repo, "
        "one line each, numbered."
    )
    messages_t3 = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": blob + "\n\n" + task3},
    ]
    st, payload, dt, ttft = chat(
        messages_t3,
        max_tokens=2048,
        timeout=900,
        stream=True,
    )
    results["turns"].append(rec_of("t3_200k_decode_list", st, payload, dt, ttft, ntok))

    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
