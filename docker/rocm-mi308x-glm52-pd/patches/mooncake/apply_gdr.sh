#!/bin/bash
# Bake true GDR Mooncake for MI308X (no C++ host-staging bounce).
# 1) MC_DISABLE_HIP_TRANSPORT so cross-node PD does not install HIP IPC
# 2) ibv_reg_mr(GPU) peermem + HipDeviceGuard (apply_gdr_peermem.py)
set -eux
ROOT="${1:-/sgl-workspace/Mooncake}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
f = Path("mooncake-transfer-engine/src/transfer_engine_impl.cpp")
s = f.read_text()
old = '''        {
            Transport* hip_transport =
                multi_transports_->installTransport("hip", nullptr);
            if (!hip_transport) {
                LOG(WARNING) << "Failed to install HIP transport "
                                "(intra-node GPU P2P unavailable)";
            } else {
                LOG(INFO) << "HIP transport installed for intra-node GPU P2P";
            }
        }'''
new = '''        {
            if (!std::getenv("MC_DISABLE_HIP_TRANSPORT") ||
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
            }
        }'''
if "MC_DISABLE_HIP_TRANSPORT" in s:
    print("transfer_engine_impl.cpp: MC_DISABLE_HIP_TRANSPORT already present")
elif old in s:
    f.write_text(s.replace(old, new, 1))
    print("transfer_engine_impl.cpp patched: MC_DISABLE_HIP_TRANSPORT")
else:
    raise SystemExit("transfer_engine_impl.cpp: HIP transport anchor not found")
PY

python3 "$HERE/apply_gdr_peermem.py" "$ROOT"
echo "[Mooncake] GDR peermem + MC_DISABLE_HIP_TRANSPORT applied (no host-staging C++)"
