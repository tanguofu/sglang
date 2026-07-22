# Find get_local_ip_auto
import sglang.srt.disaggregation.common.conn as c
import inspect
# Find the import
src = inspect.getsource(c)
for line in src.split('\n')[:30]:
    if 'get_local_ip' in line or 'import' in line and 'ip' in line.lower():
        print(line)
print("---")
# Try to find it
import sglang.srt.utils.common as uc
if hasattr(uc, 'get_local_ip_auto'):
    ip = uc.get_local_ip_auto()
    print(f"get_local_ip_auto() = {ip}")
import os
print(f"SGLANG_HOST_IP = {os.environ.get('SGLANG_HOST_IP', 'not set')}")
