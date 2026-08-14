from pathlib import Path

TARGET = Path("/sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py")
LOOPS = ("event_loop_normal_disagg_decode", "event_loop_overlap_disagg_decode")
NL = chr(10)
ANCHOR = "            self.process_input_requests(recv_reqs)" + NL + "            self.process_decode_queue()" + NL
INSERT = ANCHOR + "            # Flush queued synthetic health replies even if PD transfer work prevents a batch." + NL + "            self.maybe_send_health_check_signal()" + NL

def loop_bounds(text, loop_name):
    start = text.index(f"def {loop_name}")
    end = text.find(NL + "    def ", start + 1)
    return start, len(text) if end == -1 else end

text = TARGET.read_text()
status = []
for loop_name in LOOPS:
    start, end = loop_bounds(text, loop_name)
    block = text[start:end]
    insert_count = block.count(INSERT)
    bare_anchor_count = block.count(ANCHOR) - insert_count
    if bare_anchor_count == 1 and insert_count == 0:
        block = block.replace(ANCHOR, INSERT, 1)
        text = text[:start] + block + text[end:]
        status.append("applied")
    elif bare_anchor_count == 0 and insert_count == 1:
        status.append("already-patched")
    else:
        raise RuntimeError(f"Unexpected {loop_name} patch state: bare={bare_anchor_count}, inserted={insert_count}")

TARGET.write_text(text)
verified = TARGET.read_text()
for loop_name in LOOPS:
    start, end = loop_bounds(verified, loop_name)
    block = verified[start:end]
    if block.count(INSERT) != 1 or (block.count(ANCHOR) - block.count(INSERT)) != 0:
        raise RuntimeError(f"Verification failed for {loop_name}")
print("DECODE_PD_HEALTH_FLUSH_PATCH=" + ("applied" if "applied" in status else "already-patched"))
