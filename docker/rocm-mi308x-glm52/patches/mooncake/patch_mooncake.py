#!/usr/bin/env python3
"""Patch Mooncake C++ source for AMD ROCm RDMA compatibility.

Patch 1 — MC_DISABLE_HIP_TRANSPORT:
    Skip HIP transport install (hipIpcOpenMemHandle fails on ROCm for inter-node).

Patch 2 — MC_FORCE_DMABUF:
    Bypass isKernelDmabufSupported() check to force the dmabuf MR registration
    path (hsa_amd_portable_export_dmabuf + ibv_reg_dmabuf_mr) even when the
    kernel lacks CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY.

    Note: When SGLANG_PD_HOST_STAGING=1 is set (in the Dockerfile), SGLang
    registers host buffers with Mooncake instead of GPU memory.  Host memory
    goes through the standard ibv_reg_mr path and does NOT need dmabuf support.
    MC_FORCE_DMABUF is kept as a fallback for when host staging is not used.
"""
import os, sys


# ─── Patch 1: MC_DISABLE_HIP_TRANSPORT ───────────────────────────────────
filepath = "/sgl-workspace/Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp"
if not os.path.exists(filepath):
    print(f"ERROR: {filepath} not found")
    sys.exit(1)

with open(filepath) as f:
    content = f.read()

if "MC_DISABLE_HIP_TRANSPORT" in content:
    print("Patch 1 (MC_DISABLE_HIP_TRANSPORT): already applied")
else:
    old_block = '''        Transport* hip_transport =
                multi_transports_->installTransport("hip", nullptr);
            if (!hip_transport) {
                LOG(WARNING) << "Failed to install HIP transport "
                                "(intra-node GPU P2P unavailable)";
            } else {
                LOG(INFO) << "HIP transport installed for intra-node GPU P2P";
            }'''

    new_block = '''        if (!std::getenv("MC_DISABLE_HIP_TRANSPORT") ||
            std::string(std::getenv("MC_DISABLE_HIP_TRANSPORT")) != "1") {
            Transport* hip_transport =
                multi_transports_->installTransport("hip", nullptr);
            if (!hip_transport) {
                LOG(WARNING) << "Failed to install HIP transport "
                                "(intra-node GPU P2P unavailable)";
            } else {
                LOG(INFO) << "HIP transport installed for intra-node GPU P2P";
            }
        } else {
            LOG(INFO) << "HIP transport disabled by MC_DISABLE_HIP_TRANSPORT=1";
        }'''

    if old_block in content:
        content = content.replace(old_block, new_block)
        with open(filepath, 'w') as f:
            f.write(content)
        print("Patch 1 (MC_DISABLE_HIP_TRANSPORT): applied (exact match)")
    else:
        lines = content.split('\n')
        new_lines = []
        patched = False
        for i, line in enumerate(lines):
            if 'installTransport("hip"' in line and not patched:
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * indent + 'if (!std::getenv("MC_DISABLE_HIP_TRANSPORT") ||')
                new_lines.append(' ' * indent + '    std::string(std::getenv("MC_DISABLE_HIP_TRANSPORT")) != "1") {')
                new_lines.append(line)
                patched = True
            elif 'HIP transport installed for intra-node' in line and patched:
                new_lines.append(line)
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * (indent - 8) + '} else {')
                new_lines.append(' ' * indent + 'LOG(INFO) << "HIP transport disabled by MC_DISABLE_HIP_TRANSPORT=1";')
                new_lines.append(' ' * (indent - 8) + '}')
            else:
                new_lines.append(line)
        if patched:
            with open(filepath, 'w') as f:
                f.write('\n'.join(new_lines))
            print("Patch 1 (MC_DISABLE_HIP_TRANSPORT): applied (flexible match)")
        else:
            print("Patch 1 (MC_DISABLE_HIP_TRANSPORT): ERROR - could not find HIP transport install line!")
            sys.exit(1)

# ─── Patch 2: MC_FORCE_DMABUF ─────────────────────────────────────────────
rdma_ctx = "/sgl-workspace/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp"
if not os.path.exists(rdma_ctx):
    print(f"ERROR: {rdma_ctx} not found")
    sys.exit(1)

with open(rdma_ctx) as f:
    ctx_content = f.read()

if "MC_FORCE_DMABUF" in ctx_content:
    print("Patch 2 (MC_FORCE_DMABUF): already applied")
else:
    old_return = '''        bool ok = found[0] && found[1];
        if (!ok) {
            LOG(WARNING)
                << "Kernel lacks CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY "
                << "(p2pdma=" << found[0] << " move_notify=" << found[1]
                << "), HIP dmabuf MR registration disabled, falling back to "
                << "ibv_reg_mr() (which requires an amdgpu peermem driver). "
                << "Rebuild kernel with both options for GPU-direct RDMA.";
        }
        return ok;'''

    new_return = '''        bool ok = found[0] && found[1];
        if (!ok) {
            if (std::getenv("MC_FORCE_DMABUF") &&
                std::string(std::getenv("MC_FORCE_DMABUF")) == "1") {
                LOG(INFO) << "MC_FORCE_DMABUF=1: forcing dmabuf MR registration "
                          << "despite missing kernel P2PDMA support "
                          << "(p2pdma=" << found[0] << " move_notify=" << found[1] << ")";
                return true;
            }
            LOG(WARNING)
                << "Kernel lacks CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY "
                << "(p2pdma=" << found[0] << " move_notify=" << found[1]
                << "), HIP dmabuf MR registration disabled, falling back to "
                << "ibv_reg_mr() (which requires an amdgpu peermem driver). "
                << "Rebuild kernel with both options for GPU-direct RDMA. "
                << "Set MC_FORCE_DMABUF=1 to bypass this check.";
        }
        return ok;'''

    if old_return in ctx_content:
        ctx_content = ctx_content.replace(old_return, new_return)
        with open(rdma_ctx, 'w') as f:
            f.write(ctx_content)
        print("Patch 2 (MC_FORCE_DMABUF): applied")
    else:
        print("Patch 2 (MC_FORCE_DMABUF): ERROR - pattern not found in rdma_context.cpp!")
        sys.exit(1)

print("\nAll patches applied successfully.")
