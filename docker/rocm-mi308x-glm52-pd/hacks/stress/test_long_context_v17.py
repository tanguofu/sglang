#!/usr/bin/env python3
"""Test long context handling with v17 config."""
import requests, time

url = 'https://glm52-1tp8.jmpti.woa.com/v1/chat/completions'
headers = {'Authorization': 'Bearer sk-46faecc9d0bc4dcd9db6a15c73ae91c8', 'Content-Type': 'application/json'}

print("--- Long Context Test (streaming, no-think) ---")
for n_repeats in [10, 50, 100, 200, 500, 1000]:
    prompt = 'The quick brown fox jumps over the lazy dog. ' * n_repeats
    payload = {
        'model': 'glm-5.2',
        'messages': [{'role': 'user', 'content': f'Count the number of times the word fox appears in this text:\n{prompt}\n\nAnswer with just the number.'}],
        'max_tokens': 20,
        'temperature': 0,
        'chat_template_kwargs': {'enable_thinking': False},
        'stream': True,
    }
    start = time.time()
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=300, stream=True)
        wall = time.time() - start
        if r.status_code == 200:
            content = ''
            for line in r.iter_lines():
                if line and line.startswith(b'data: ') and b'[DONE]' not in line:
                    import json
                    chunk = json.loads(line[6:])
                    delta = chunk.get('choices', [{}])[0].get('delta', {})
                    if delta.get('content'):
                        content += delta['content']
            usage_line = ''
            # Get final usage from last chunk
            print(f'  repeats={n_repeats:5d} | HTTP=200 | wall={wall:.2f}s | answer={content[:50]}')
        else:
            print(f'  repeats={n_repeats:5d} | HTTP={r.status_code} | wall={wall:.2f}s | error={r.text[:100]}')
    except Exception as e:
        wall = time.time() - start
        print(f'  repeats={n_repeats:5d} | ERROR | wall={wall:.2f}s | {str(e)[:100]}')

print("\n--- Non-streaming long context ---")
for n_repeats in [10, 100, 500]:
    prompt = 'The quick brown fox jumps over the lazy dog. ' * n_repeats
    payload = {
        'model': 'glm-5.2',
        'messages': [{'role': 'user', 'content': f'How many times does "fox" appear? Just the number.\n{prompt}'}],
        'max_tokens': 20,
        'temperature': 0,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    start = time.time()
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=300)
        wall = time.time() - start
        if r.status_code == 200:
            d = r.json()
            content = d['choices'][0]['message']['content']
            usage = d['usage']
            print(f'  repeats={n_repeats:5d} | prompt_toks={usage["prompt_tokens"]:6d} | HTTP=200 | wall={wall:.2f}s | answer={content[:30]}')
        else:
            print(f'  repeats={n_repeats:5d} | HTTP={r.status_code} | wall={wall:.2f}s | error={r.text[:100]}')
    except Exception as e:
        wall = time.time() - start
        print(f'  repeats={n_repeats:5d} | ERROR | wall={wall:.2f}s | {str(e)[:100]}')
