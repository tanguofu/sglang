#!/usr/bin/env python3
"""Fix _bootstrap_addr in disaggregation/decode.py to handle list-type bootstrap_host.

Problem: the PD router (sgl-model-gateway) sends bootstrap_host as a LIST
(e.g. ["NODE_PREFILL_0_IP"]) for 1p1d deployments. NetworkAddress.__post_init__
calls self.host.startswith("[") which raises AttributeError on a list,
crashing all TP ranks on every PD transfer attempt.

Fix: normalize list -> str (take first element) in _bootstrap_addr before
passing to NetworkAddress. Keeps NetworkAddress pure.

Idempotent.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/port-post1/base/srt_disaggregation_decode.py")
src = path.read_text()
changed = []

marker = "# FIX(bootstrap-list): normalize list-type bootstrap_host"

if marker in src:
    changed.append("bootstrap-list: skipped")
else:
    old = (
        "def _bootstrap_addr(req: Req) -> str:\n"
        "    # FIXME: make a property of a req\n"
        "    return NetworkAddress(req.bootstrap_host, req.bootstrap_port).to_host_port_str()\n"
    )
    new = (
        "def _bootstrap_addr(req: Req) -> str:\n"
        "    # FIXME: make a property of a req\n"
        "    # FIX(bootstrap-list): normalize list-type bootstrap_host\n"
        "    # The PD router may send bootstrap_host as [\"ip\"] (list) for 1p1d.\n"
        "    host = req.bootstrap_host\n"
        "    if isinstance(host, list):\n"
        "        host = host[0] if host else None\n"
        "    port = req.bootstrap_port\n"
        "    if isinstance(port, list):\n"
        "        port = port[0] if port else None\n"
        "    return NetworkAddress(host, port).to_host_port_str()\n"
    )
    assert old in src, "bootstrap-list: _bootstrap_addr anchor not found"
    src = src.replace(old, new, 1)
    changed.append("bootstrap-list: applied")

path.write_text(src)
print(f"[ok] {path}: " + ", ".join(changed))
