import inspect
from sglang.srt.disaggregation.mooncake.conn import get_mooncake_transfer_engine
src = inspect.getsource(get_mooncake_transfer_engine)
print(src[:3000])
