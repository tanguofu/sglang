"""
EAGLE reproduction test (with auth).
Tests EAGLE-enabled config to reproduce decode coredump.
"""
import sys
import time
import json
import urllib.request
import urllib.error

ROUTER = "http://localhost:30001"
WORKER_W1 = "http://21.151.225.152:30000"
WORKER_W2 = "http://21.151.225.172:30000"
API_KEY = "sk-46faecc9d0bc4dcd9db6a15c73ae91c8"
MODEL = "glm-5.2"

def chat(prompt, max_tokens=512, temperature=0.7, url=ROUTER):
    """Send chat completion request."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
            return {
                "ok": True,
                "content": data["choices"][0]["message"]["content"][:300],
                "usage": data.get("usage", {}),
                "finish_reason": data["choices"][0].get("finish_reason"),
            }
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}", "body": e.read()[:200].decode("utf-8", errors="ignore")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Large prompt (~1024 tokens prefill — known to trigger EAGLE coredump)
LARGE_PROMPT = """
请详细分析以下技术文档的内容,并总结关键要点:

人工智能领域近年来取得了显著进展,特别是在大语言模型(LLM)方面。大语言模型通过在海量文本数据上训练,学习语言的统计规律和语义知识,展现出强大的自然语言理解和生成能力。从早期的统计语言模型,到神经语言模型,再到如今的Transformer架构大模型,技术发展经历了多个阶段。

Transformer架构的核心创新在于自注意力机制(self-attention mechanism),它允许模型在处理每个位置时同时关注输入序列中的所有其他位置,从而捕获长距离依赖关系。这种并行计算特性也使得Transformer比传统的RNN/LSTM更适合在大规模数据上并行训练。多头注意力(multi-head attention)进一步增强了模型的表达能力,使其能够在不同子空间中学习不同的关注模式。

大语言模型的训练通常分为预训练(pre-training)和微调(fine-tuning)两个阶段。预训练阶段,模型在大规模无标注文本上进行自监督学习,目标是预测下一个token或掩码token。通过这种方式,模型学习到了丰富的语言知识和世界知识。微调阶段则使用有标注的数据对模型进行特定任务的训练,如对话、问答、代码生成等。指令微调(instruction tuning)和人类反馈强化学习(RLHF)是当前主流的微调方法。

模型规模的增长是大语言模型性能提升的关键因素之一。从GPT-3的1750亿参数,到PaLM的5400亿参数,再到更大的万亿参数模型,模型规模的增加带来了显著的性能提升,这种现象被称为"scaling laws"。然而,模型规模的增加也带来了训练和推理的计算挑战,需要分布式训练技术、混合精度训练、梯度检查点等优化手段。

推理优化是大语言模型落地的关键技术。由于自回归生成需要逐token生成,推理延迟较高。各种加速技术应运而生:KV Cache缓存已生成的key-value对避免重复计算;量化技术(如INT8、FP8)降低显存占用和计算开销;投机采样(speculative decoding)通过小模型生成候选token再由大模型验证,显著加速推理;FlashAttention等硬件感知优化则减少了内存访问开销。

分布式推理是将大模型部署到多GPU/多节点的核心技术。张量并行(Tensor Parallelism)将模型参数切分到多个GPU上;流水线并行(Pipeline Parallelism)将模型按层切分到不同设备;专家并行(Expert Parallelism)则针对MoE模型将不同专家分布到不同设备。vLLM、SGLang、TensorRT-LLM等推理框架提供了这些并行策略的工程实现。

请基于以上内容,深入分析:
1. Transformer架构相比传统RNN的优势是什么?
2. 大语言模型训练的两个主要阶段是什么?各自的目标是什么?
3. 推理优化技术有哪些?各自解决什么问题?
4. 分布式推理的并行策略有哪些?适用场景是什么?
5. 投机采样的原理是什么?如何加速推理?
""".strip()

print(f"=== Test 1: Small request (warm-up, baseline) ===")
t0 = time.time()
result = chat("hello, please say hi briefly", max_tokens=30, temperature=0.0)
elapsed = time.time() - t0
print(f"Elapsed: {elapsed:.2f}s ok={result.get('ok')}")
if result.get("ok"):
    print(f"  finish: {result.get('finish_reason')}, usage: {result.get('usage')}")
    print(f"  content: {result.get('content')[:100]}")
else:
    print(f"  error: {result.get('error')} body: {result.get('body')}")
print()

time.sleep(2)

print(f"=== Test 2: Large prompt (1024+ tokens) via router — EAGLE enabled ===")
print(f"Prompt size: {len(LARGE_PROMPT)} chars (~{len(LARGE_PROMPT)//3} tokens)")
t0 = time.time()
result = chat(LARGE_PROMPT, max_tokens=512, temperature=0.7)
elapsed = time.time() - t0
print(f"Elapsed: {elapsed:.2f}s ok={result.get('ok')}")
if result.get("ok"):
    print(f"  finish: {result.get('finish_reason')}")
    print(f"  usage: {result.get('usage')}")
    print(f"  content (first 300 chars): {result.get('content')}")
else:
    print(f"  error: {result.get('error')} body: {result.get('body')}")
print()

time.sleep(3)

print(f"=== Test 3: Health check (detect zombie) ===")
t0 = time.time()
result3 = chat("hi", max_tokens=10, temperature=0.0)
elapsed = time.time() - t0
print(f"Elapsed: {elapsed:.2f}s ok={result3.get('ok')}")
if result3.get("ok"):
    print(f"  content: {result3.get('content')[:100]}")
else:
    print(f"  error: {result3.get('error')} body: {result3.get('body')}")
print()

print(f"=== Test 4: Direct hit W1 (bypass router) ===")
t0 = time.time()
result4 = chat("hello", max_tokens=20, temperature=0.0, url=WORKER_W1)
elapsed = time.time() - t0
print(f"W1 Elapsed: {elapsed:.2f}s ok={result4.get('ok')} content: {result4.get('content') if result4.get('ok') else result4.get('error')}")
print()

print(f"=== Test 5: Direct hit W2 (bypass router) ===")
t0 = time.time()
result5 = chat("hello", max_tokens=20, temperature=0.0, url=WORKER_W2)
elapsed = time.time() - t0
print(f"W2 Elapsed: {elapsed:.2f}s ok={result5.get('ok')} content: {result5.get('content') if result5.get('ok') else result5.get('error')}")
