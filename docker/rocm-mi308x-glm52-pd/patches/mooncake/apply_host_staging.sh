#!/bin/bash
set -eux
cd /sgl-workspace/Mooncake

# Patch 1: Add staging fields to MemoryRegionMeta in rdma_context.h
python3 -c "
f = 'mooncake-transfer-engine/include/transport/rdma_transport/rdma_context.h'
s = open(f).read()
old = 'struct ibv_mr *mr;\n};'
new = '''struct ibv_mr *mr;
    void *staging_host_buf = nullptr;
    void *staging_gpu_addr = nullptr;
    size_t staging_length = 0;
};'''
if old in s:
    s = s.replace(old, new, 1)
    open(f, 'w').write(s)
    print('Header patched: staging fields added')
else:
    print('Header already patched or pattern not found')
"

# Patch 2: Add host staging to #else block in rdma_context.cpp
python3 -c "
f = 'mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp'
s = open(f).read()
old = '''#else
    mrMeta.addr = addr;
    mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);
#endif'''
new = '''#else
    {
        hipPointerAttribute_t elseAttr{};
        hipError_t elseRes = hipPointerGetAttributes(&elseAttr, addr);
        if (elseRes == hipSuccess && elseAttr.type == hipMemoryTypeDevice) {
            HipDeviceGuard dev_guard(elseAttr.device);
            if (!dev_guard.set_ok()) { return ERR_CONTEXT; }
            void* host_buf = nullptr;
            if (hipMallocHost(&host_buf, length) != hipSuccess) { return ERR_CONTEXT; }
            if (hipMemcpy(host_buf, addr, length, hipMemcpyDeviceToHost) != hipSuccess) {
                (void)hipFreeHost(host_buf); return ERR_CONTEXT; }
            mrMeta.addr = addr;
            mrMeta.staging_host_buf = host_buf;
            mrMeta.staging_gpu_addr = addr;
            mrMeta.staging_length = length;
            mrMeta.mr = ibv_reg_mr(pd_, host_buf, length, access);
            LOG(INFO) << \"RDMA host staging: GPU=\" << addr << \" host=\" << host_buf << \" len=\" << length;
        } else {
            mrMeta.addr = addr;
            mrMeta.mr = ibv_reg_mr(pd_, addr, length, access);
        }
    }
#endif'''
if old in s:
    s = s.replace(old, new, 1)
    open(f, 'w').write(s)
    print('rdma_context.cpp patched: host staging in #else block')
else:
    print('rdma_context.cpp already patched or pattern not found')
"

# Patch 3: Add MC_DISABLE_HIP_TRANSPORT to transfer_engine_impl.cpp
python3 -c "
f = 'mooncake-transfer-engine/src/transfer_engine_impl.cpp'
s = open(f).read()
old = '''        {
            Transport* hip_transport =
                multi_transports_->installTransport(\"hip\", nullptr);'''
new = '''        {
          if (!std::getenv(\"MC_DISABLE_HIP_TRANSPORT\") ||
              std::string(std::getenv(\"MC_DISABLE_HIP_TRANSPORT\")) != \"1\") {
            Transport* hip_transport =
                multi_transports_->installTransport(\"hip\", nullptr);'''
if old in s:
    s = s.replace(old, new, 1)
    # Also add the closing brace and else block
    old2 = '''            LOG(INFO) << \"HIP transport installed for intra-node GPU P2P\";
        }'''
    new2 = '''            LOG(INFO) << \"HIP transport installed for intra-node GPU P2P\";
          } else {
            LOG(INFO) << \"HIP transport disabled by MC_DISABLE_HIP_TRANSPORT=1\";
          }
        }'''
    if old2 in s:
        s = s.replace(old2, new2, 1)
    open(f, 'w').write(s)
    print('transfer_engine_impl.cpp patched: MC_DISABLE_HIP_TRANSPORT')
else:
    print('transfer_engine_impl.cpp already patched or pattern not found')
"
