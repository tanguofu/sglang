#!/usr/bin/env python3
"""NCCL RDMA connectivity test between two nodes.

Run on both nodes with proper env vars. Tests all-reduce across 16 GPUs.
"""
import os, torch, torch.distributed as dist

dist.init_process_group(
    backend="nccl",
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    rank=int(os.environ['RANK']),
    world_size=int(os.environ['WORLD_SIZE']),
)
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get('LOCAL_RANK', rank % int(os.environ.get('NPROC_PER_NODE', 8))))
torch.cuda.set_device(local_rank)

t = torch.ones(1024, device="cuda", dtype=torch.float32) * (rank + 1)
dist.all_reduce(t, op=dist.ReduceOp.SUM)
expected = sum(range(1, world_size + 1))
got = t[0].item()
ok = abs(got - expected) < 0.5
print(f"[rank {rank}/{world_size} local {local_rank}] all_reduce result={got:.1f} expected={expected} {'OK' if ok else 'FAIL'}", flush=True)

if rank == 0:
    print("\nNCCL RDMA TEST PASSED" if ok else "\nNCCL RDMA TEST FAILED", flush=True)
dist.barrier()
dist.destroy_process_group()
