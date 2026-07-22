import time, json, urllib.request, ssl

URL = 'https://glm52-2tp8.jmpti.woa.com/v1/chat/completions'
API_KEY = 'sk-46faecc9d0bc4dcd9db6a15c73ae91c8'
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def ttft_test(label, num_tokens, max_tokens=10, enable_thinking=False, repeat=3, stream=True):
    content = 'Test sentence. ' * (num_tokens // 2) + 'Reply hi'
    for i in range(repeat):
        data = json.dumps({
            'model': 'glm-5.2',
            'messages': [{'role': 'user', 'content': content}],
            'max_tokens': max_tokens,
            'stream': stream,
            'chat_template_kwargs': {'enable_thinking': enable_thinking}
        }).encode()
        req = urllib.request.Request(URL, data=data, headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        })
        start = time.time()
        first_byte = None
        try:
            resp = urllib.request.urlopen(req, timeout=120, context=ctx)
            if stream:
                for line in resp:
                    if line.startswith(b'data:'):
                        first_byte = time.time()
                        break
                total = time.time() - start
                ttft = (first_byte - start) if first_byte else None
                print(f'{label} iter={i} TTFT={ttft:.3f}s total={total:.3f}s' if ttft else f'{label} iter={i} NO_DATA total={total:.3f}s')
            else:
                body = resp.read()
                total = time.time() - start
                print(f'{label} iter={i} total={total:.3f}s (non-stream)')
        except Exception as e:
            total = time.time() - start
            print(f'{label} iter={i} ERROR={type(e).__name__}:{e} total={total:.3f}s')
        time.sleep(1)

print('=== Gateway TTFT Test (HTTPS) ===')
print()
print('--- Test 1: warmup (13 tokens) ---')
ttft_test('warmup', 13, repeat=1)
print()
print('--- Test 2: 1K tokens x3 ---')
ttft_test('1K', 1000, repeat=3)
print()
print('--- Test 3: 3K tokens x3 ---')
ttft_test('3K', 3000, repeat=3)
print()
print('--- Test 4: 1K again (after warm) ---')
ttft_test('1K-warm', 1000, repeat=2)
print()
print('--- Test 5: 5K tokens x1 ---')
ttft_test('5K', 5000, repeat=1)
