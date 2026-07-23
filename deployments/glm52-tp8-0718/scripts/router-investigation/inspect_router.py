import sglang_router.launch_router as m
print('Module file:', m.__file__)
import inspect
src = inspect.getsource(m)
import re
routes = re.findall(r'@(app|router|router_obj)\.(get|post|put|delete)\(["\']([^"\']+)["\']', src)
print('Routes found:')
for r in routes:
    print(f'  {r[1].upper()} {r[2]}')
print('---')
# Look for messages/responses keywords
for line in src.splitlines():
    if any(k in line.lower() for k in ['messages', 'responses', '/v1/', 'anthropic', 'openai']):
        print(line.strip()[:150])
