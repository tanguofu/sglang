import os
# Check all mooncake-related env vars
for k, v in sorted(os.environ.items()):
    if any(x in k.upper() for x in ['MC_', 'MOONCAKE', 'RDMA', 'IB_', 'NCCL_IB', 'SGLANG_PD', 'SGLANG_HOST']):
        print(f'{k}={v}')
