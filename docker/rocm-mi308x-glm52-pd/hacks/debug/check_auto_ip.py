from sglang.srt.utils import get_local_ip_auto
ip = get_local_ip_auto()
print(f"get_local_ip_auto() = {ip}")
import os
print(f"SGLANG_HOST_IP = {os.environ.get('SGLANG_HOST_IP', 'not set')}")
