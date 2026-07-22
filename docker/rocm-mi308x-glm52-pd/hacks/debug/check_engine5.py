import inspect
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import MooncakeTransferEngine
src = inspect.getsource(MooncakeTransferEngine.initialize)
print(src[:3000])
