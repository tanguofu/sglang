#!/usr/bin/env python3
"""Patch Mooncake C++ source to add MC_DISABLE_HIP_TRANSPORT env var."""
import os, sys

filepath = "/sgl-workspace/Mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp"
if not os.path.exists(filepath):
    print(f"ERROR: {filepath} not found")
    sys.exit(1)

with open(filepath) as f:
    content = f.read()

if "MC_DISABLE_HIP_TRANSPORT" in content:
    print("Already patched, skipping.")
    sys.exit(0)

# Find the line that installs HIP transport
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
    print("Patched successfully (exact match).")
    sys.exit(0)
else:
    # Try a more flexible approach: find the line and insert before it
    lines = content.split('\n')
    new_lines = []
    patched = False
    for i, line in enumerate(lines):
        if 'installTransport("hip"' in line and not patched:
            # Insert the if-check before this line
            indent = len(line) - len(line.lstrip())
            new_lines.append(' ' * indent + 'if (!std::getenv("MC_DISABLE_HIP_TRANSPORT") ||')
            new_lines.append(' ' * indent + '    std::string(std::getenv("MC_DISABLE_HIP_TRANSPORT")) != "1") {')
            new_lines.append(line)
            patched = True
        elif 'HIP transport installed for intra-node' in line and patched:
            new_lines.append(line)
            # Add closing brace after the log line
            indent = len(line) - len(line.lstrip())
            new_lines.append(' ' * (indent - 8) + '} else {')
            new_lines.append(' ' * indent + 'LOG(INFO) << "HIP transport disabled by MC_DISABLE_HIP_TRANSPORT=1";')
            new_lines.append(' ' * (indent - 8) + '}')
        else:
            new_lines.append(line)
    
    if patched:
        with open(filepath, 'w') as f:
            f.write('\n'.join(new_lines))
        print("Patched successfully (flexible match).")
        sys.exit(0)
    else:
        print("ERROR: Could not find HIP transport install line!")
        sys.exit(1)
