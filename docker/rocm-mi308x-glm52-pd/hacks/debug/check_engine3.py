import inspect
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    get_mooncake_transfer_engine as _get_engine,
    MooncakeTransferEngine,
)
# Find the initialization
src = inspect.getsource(MooncakeTransferEngine)
# Look for local_ip, ib_device, protocol
for i, line in enumerate(src.split('\n')):
    if any(k in line.lower() for k in ['local_ip', 'ib_device', 'protocol', 'gid_index', 'mc_', 'mooncake_protocol', 'pd_host']):
        print(f'{i}: {line}')
