from pathlib import Path

TARGET = Path("/sgl-workspace/sglang/python/sglang/srt/managers/overlap_utils.py")

# v0.5.17 inserts a `_DEBUG_ASSERT` consume-once block between
# `if self.publish_ready is not None:` and `if _is_hip:`, so matching the
# full block fails.  Match only the _is_hip branch — it appears exactly once
# in the file and is stable across v0.5.16 / v0.5.17.
#
# NOTE: NEW (`            self.publish_ready.wait()\n`, 12 spaces) is a substring
# of OLD's else branch (`                self.publish_ready.wait()\n`, 16 spaces),
# so we cannot use new_count for idempotency detection.  Rely solely on
# old_count: 1 → apply, 0 → already-patched.
OLD = """            if _is_hip:
                # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
                self.publish_ready.synchronize()
            else:
                self.publish_ready.wait()
"""
NEW = """            self.publish_ready.wait()
"""
text = TARGET.read_text()
old_count = text.count(OLD)
if old_count == 1:
    text = text.replace(OLD, NEW, 1)
    TARGET.write_text(text)
    status = "applied"
elif old_count == 0:
    status = "already-patched"
else:
    raise RuntimeError(f"Unexpected overlap patch state: old={old_count}")
verified = TARGET.read_text()
if verified.count(OLD) != 0:
    raise RuntimeError("Overlap HIP wait patch verification failed: OLD still present")
# old_count==0 used to mean "already patched" even when the HIP synchronize()
# branch was just a whitespace mismatch and still in the tree. Fail closed.
if "self.publish_ready.synchronize()" in verified:
    raise RuntimeError("Overlap HIP wait: publish_ready.synchronize() still present")
if "Event.wait() regresses TPOT" in verified:
    raise RuntimeError("Overlap HIP wait: MI355 synchronize workaround still present")
print(f"OVERLAP_HIP_WAIT_PATCH={status}")
