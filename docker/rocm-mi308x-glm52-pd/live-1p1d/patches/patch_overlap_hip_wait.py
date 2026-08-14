from pathlib import Path

TARGET = Path("/sgl-workspace/sglang/python/sglang/srt/managers/overlap_utils.py")
OLD = """        if self.publish_ready is not None:
            if _is_hip:
                # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
                self.publish_ready.synchronize()
            else:
                self.publish_ready.wait()
"""
NEW = """        if self.publish_ready is not None:
            self.publish_ready.wait()
"""
text = TARGET.read_text()
old_count = text.count(OLD)
new_count = text.count(NEW)
if old_count == 1 and new_count == 0:
    text = text.replace(OLD, NEW, 1)
    TARGET.write_text(text)
    status = "applied"
elif old_count == 0 and new_count == 1:
    status = "already-patched"
else:
    raise RuntimeError(f"Unexpected overlap patch state: old={old_count}, new={new_count}")
verified = TARGET.read_text()
if verified.count(NEW) != 1 or verified.count(OLD) != 0:
    raise RuntimeError("Overlap HIP wait patch verification failed")
print(f"OVERLAP_HIP_WAIT_PATCH={status}")
