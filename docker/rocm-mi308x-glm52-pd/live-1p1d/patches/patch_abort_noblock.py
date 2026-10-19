from pathlib import Path

TARGET = Path("/sgl-workspace/sglang/python/sglang/srt/disaggregation/common/conn.py")
NL = chr(10)
OLD = "                        ]" + NL + "                    )"
NEW = "                        ]," + NL + "                        flags=zmq.NOBLOCK," + NL + "                    )"
text = TARGET.read_text()
old_count = text.count(OLD)
new_count = text.count(NEW)
if old_count == 1 and new_count == 0:
    TARGET.write_text(text.replace(OLD, NEW, 1))
    status = "applied"
elif old_count == 0 and new_count == 1:
    status = "already-patched"
else:
    raise RuntimeError(f"Unexpected abort-notification patch state: old={old_count}, new={new_count}")
verified = TARGET.read_text()
if verified.count(NEW) != 1 or verified.count(OLD) != 0:
    raise RuntimeError("Abort NOBLOCK patch verification failed")
if "import zmq" not in verified:
    raise RuntimeError("zmq import missing")
print(f"ABORT_NOBLOCK_PATCH={status}")
