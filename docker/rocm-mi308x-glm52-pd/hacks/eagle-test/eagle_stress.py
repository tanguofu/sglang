"""
Aggressive EAGLE stress test — concurrent requests via router.
Goal: trigger TP rank divergence in greedy EAGLE verify path.
"""
import sys
import time
import json
import urllib.request
import urllib.error
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROUTER = "http://localhost:30001"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"

def chat(prompt, max_tokens=512, temperature=0.7, request_id=0):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{ROUTER}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
            elapsed = time.time() - t0
            return {
                "id": request_id,
                "ok": True,
                "elapsed": elapsed,
                "finish": data["choices"][0].get("finish_reason"),
                "completion_tokens": data.get("usage", {}).get("completion_tokens"),
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
            }
    except urllib.error.HTTPError as e:
        return {"id": request_id, "ok": False, "elapsed": time.time() - t0,
                "error": f"HTTP {e.code}", "body": e.read()[:150].decode("utf-8", errors="ignore")}
    except Exception as e:
        return {"id": request_id, "ok": False, "elapsed": time.time() - t0, "error": str(e)}

# Medium prompts — varied to hit both workers
PROMPTS = [
    "请详细解释深度学习中注意力机制的原理,包括自注意力和多头注意力的区别。",
    "Explain the difference between Tensor Parallelism and Pipeline Parallelism in distributed LLM inference, with concrete examples of when to use each.",
    "请分析大语言模型推理优化的关键技术,包括KV Cache、量化、投机采样、FlashAttention。",
    "What are the key challenges in training trillion-parameter language models? Discuss memory, compute, and communication bottlenecks.",
    "请对比vLLM、SGLang、TensorRT-LLM三个推理框架的架构特点和适用场景。",
    "Describe the transformer architecture in detail. What are the key innovations compared to RNN/LSTM?",
    "请解释MoE(Mixture of Experts)模型的工作原理,以及专家并行的实现方式。",
    "What is speculative decoding? How does EAGLE differ from traditional draft-target speculative sampling?",
    "请分析RLHF(基于人类反馈的强化学习)在语言模型对齐中的作用和挑战。",
    "Explain the scaling laws for large language models. How do loss, compute, and dataset size relate?",
] * 3  # 30 prompts total

print(f"=== Stress test: {len(PROMPTS)} concurrent requests via router (EAGLE enabled) ===")
print(f"Each prompt: ~100-200 tokens, max_tokens=400, temperature=0.7")
print()

results = []
t_start = time.time()

# Launch 8 concurrent requests at a time
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(chat, p, 400, 0.7, i): i for i, p in enumerate(PROMPTS)}
    for f in as_completed(futures):
        r = f.result()
        results.append(r)
        status = "OK" if r["ok"] else f"FAIL: {r.get('error','?')} {r.get('body','')[:80]}"
        print(f"  req {r['id']:2d}: {r['elapsed']:6.2f}s {status} "
              f"finish={r.get('finish','-')} ct={r.get('completion_tokens','-')}")

t_total = time.time() - t_start
ok_count = sum(1 for r in results if r["ok"])
fail_count = sum(1 for r in results if not r["ok"])

print()
print(f"=== Summary ===")
print(f"Total: {len(results)}, OK: {ok_count}, FAIL: {fail_count}")
print(f"Total elapsed: {t_total:.2f}s")
if ok_count:
    ok_times = [r["elapsed"] for r in results if r["ok"]]
    print(f"OK latency: min={min(ok_times):.2f}s, max={max(ok_times):.2f}s, avg={sum(ok_times)/len(ok_times):.2f}s")
