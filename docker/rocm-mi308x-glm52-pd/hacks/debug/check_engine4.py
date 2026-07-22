import inspect
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import MooncakeTransferEngine
src = inspect.getsource(MooncakeTransferEngine.__init__)
print(src[:3500])
