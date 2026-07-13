#!/usr/bin/env python3
"""Add debug logging to the PD prefill event loop to pinpoint the
/v1/responses hang (run_batch forward pass vs send_kv_chunk KV transfer)."""
import sys

P = "/sgl-workspace/sglang/python/sglang/srt/disaggregation/prefill.py"
s = open(P).read()
if "[DBG]" in s:
    print("[OK] prefill debug logging already present")
    sys.exit(0)

# 1. Around run_batch + process_batch_result in event_loop_normal_disagg_prefill
old1 = (
    "                result = self.run_batch(batch)\n"
    "                self.process_batch_result(batch, result)\n"
)
new1 = (
    "                print(f'[DBG] run_batch start rid={batch.reqs[0].rid}', flush=True)\n"
    "                result = self.run_batch(batch)\n"
    "                print('[DBG] run_batch done', flush=True)\n"
    "                self.process_batch_result(batch, result)\n"
    "                print('[DBG] process_batch_result done', flush=True)\n"
)
assert old1 in s, "run_batch anchor not found"
s = s.replace(old1, new1, 1)

# 2. Around send_kv_chunk in process_batch_result_disagg_prefill
old2 = "            self.send_kv_chunk(req, last_chunk=True)\n"
new2 = (
    "            print(f'[DBG] send_kv_chunk start rid={req.rid}', flush=True)\n"
    "            self.send_kv_chunk(req, last_chunk=True)\n"
    "            print(f'[DBG] send_kv_chunk done rid={req.rid}', flush=True)\n"
)
assert old2 in s, "send_kv_chunk anchor not found"
s = s.replace(old2, new2, 1)

open(P, "w").write(s)
print("[OK] prefill debug logging added (run_batch + send_kv_chunk)")
