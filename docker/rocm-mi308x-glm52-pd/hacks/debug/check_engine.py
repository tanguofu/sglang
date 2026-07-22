import inspect
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
src = inspect.getsource(MooncakeKVManager.init_engine)
print(src)
