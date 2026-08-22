#!/usr/bin/env python3
"""Coerce PD bootstrap_room/host/port lists to scalars.

Codex /v1/responses on this 1P1D stack injects bootstrap_room as a list.
MooncakeKVSender then does `if room not in request_status` and all TP ranks
die with TypeError: unhashable type: 'list' (prefill exit 137).

Idempotent. Safe to run on prefill and decode. Chat completions already
uses a scalar room and is unchanged.
"""
from pathlib import Path

HELPER = '''
def as_bootstrap_room(room):
    """Coerce PD-router bootstrap_room to a hashable scalar."""
    if isinstance(room, (list, tuple)):
        return room[0] if room else None
    return room

'''

UNWRAP_SINGLE = '''        if isinstance(self.bootstrap_host, list):
            self.bootstrap_host = self.bootstrap_host[0] if self.bootstrap_host else None
        if isinstance(self.bootstrap_port, list):
            self.bootstrap_port = self.bootstrap_port[0] if self.bootstrap_port else None
        if isinstance(self.bootstrap_room, list):
            self.bootstrap_room = self.bootstrap_room[0] if self.bootstrap_room else None
'''


def patch_common_conn(path: Path) -> str:
    if not path.exists():
        return f"{path}: missing"
    text = path.read_text()
    changed = []
    if "def as_bootstrap_room(" not in text:
        anchor = "logger = logging.getLogger(__name__)\n"
        assert anchor in text, f"{path}: logger anchor missing"
        text = text.replace(anchor, anchor + "\n" + HELPER, 1)
        changed.append("helper")
    old_sender = "        self.kv_mgr = mgr\n        self.bootstrap_room = bootstrap_room\n        self.aux_index = None"
    new_sender = "        self.kv_mgr = mgr\n        self.bootstrap_room = as_bootstrap_room(bootstrap_room)\n        self.aux_index = None"
    if old_sender in text:
        text = text.replace(old_sender, new_sender, 1)
        changed.append("sender")
    elif "self.bootstrap_room = as_bootstrap_room(bootstrap_room)" in text:
        changed.append("sender-skipped")
    old_recv = "        self.bootstrap_room = bootstrap_room\n        self.bootstrap_addr = bootstrap_addr\n        self.kv_mgr = mgr"
    new_recv = "        self.bootstrap_room = as_bootstrap_room(bootstrap_room)\n        self.bootstrap_addr = bootstrap_addr\n        self.kv_mgr = mgr"
    if old_recv in text:
        text = text.replace(old_recv, new_recv, 1)
        changed.append("receiver")
    elif "self.bootstrap_room = as_bootstrap_room(bootstrap_room)" in text:
        changed.append("receiver-skipped")
    path.write_text(text)
    return f"{path.name}: {', '.join(changed) or 'no-op'}"


def patch_io_struct(path: Path) -> str:
    if not path.exists():
        return f"{path}: missing"
    text = path.read_text()
    if "isinstance(self.bootstrap_room, list)" in text and "self.bootstrap_room = self.bootstrap_room[0]" in text:
        return f"{path.name}: skipped"
    old = "        if not self.token_ids_logprob:  # covers both None and []\n            self.token_ids_logprob = None\n"
    assert old in text, f"{path}: _normalize_single_inputs anchor missing"
    text = text.replace(old, old + UNWRAP_SINGLE, 1)
    path.write_text(text)
    return f"{path.name}: applied"


def patch_tokenizer(path: Path) -> str:
    if not path.exists():
        return f"{path}: missing"
    text = path.read_text()
    if "isinstance(bootstrap_room, (list, tuple))" in text:
        return f"{path.name}: skipped"
    old = "            bootstrap_room = obj.bootstrap_room\n            if (\n                bootstrap_room is None\n"
    new = (
        "            bootstrap_room = obj.bootstrap_room\n"
        "            if isinstance(bootstrap_room, (list, tuple)):\n"
        "                bootstrap_room = bootstrap_room[0] if bootstrap_room else None\n"
        "            if (\n"
        "                bootstrap_room is None\n"
    )
    assert old in text, f"{path}: tokenizer bootstrap_room anchor missing"
    path.write_text(text.replace(old, new, 1))
    return f"{path.name}: applied"


def main():
    root = Path("/sgl-workspace/sglang/python/sglang/srt")
    reports = [
        patch_common_conn(root / "disaggregation/common/conn.py"),
        patch_io_struct(root / "managers/io_struct.py"),
        patch_tokenizer(root / "managers/tokenizer_manager.py"),
    ]
    print("BOOTSTRAP_ROOM_SCALAR=" + "; ".join(reports))


if __name__ == "__main__":
    main()
