import inspect
from sglang.srt.disaggregation.common.conn import CommonKVManager
src = inspect.getsource(CommonKVManager.__init__)
# Find local_ip related lines
for i, line in enumerate(src.split('\n')):
    if 'local_ip' in line.lower() or 'host_ip' in line.lower() or 'SGLANG_HOST_IP' in line:
        print(f'{i}: {line}')
print("---")
# Check the init_engine method
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
src2 = inspect.getsource(MooncakeKVManager.init_engine)
for i, line in enumerate(src2.split('\n')):
    if 'local_ip' in line.lower() or 'host_ip' in line.lower() or 'device' in line.lower() or 'ib_device' in line.lower():
        print(f'init_engine {i}: {line}')
