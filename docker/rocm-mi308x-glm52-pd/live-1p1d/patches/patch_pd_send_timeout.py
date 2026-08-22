from pathlib import Path

NL = chr(10)
applied = []

# --- Part A: common/conn.py — bounded send on the bootstrap-server classmethod socket factory ---
common = Path("/sgl-workspace/sglang/python/sglang/srt/disaggregation/common/conn.py")
text = common.read_text()
A_OLD = "                sock.setsockopt(zmq.LINGER, 0)" + NL + "                sock.connect(endpoint)"
A_NEW = "                sock.setsockopt(zmq.LINGER, 0)" + NL + "                sock.setsockopt(zmq.SNDTIMEO, 5000)" + NL + "                sock.connect(endpoint)"
a_old, a_new = text.count(A_OLD), text.count(A_NEW)
if a_old == 1 and a_new == 0:
    text = text.replace(A_OLD, A_NEW, 1)
    applied.append("sndtimeo")
elif a_old == 0 and a_new == 1:
    pass
elif a_old == 0 and a_new == 0:
    print("PD_SEND_TIMEOUT: common SNDTIMEO already present or factory differs, skip")
else:
    raise RuntimeError(f"common SNDTIMEO anchor mismatch: old={a_old} new={a_new}")
common.write_text(text)

# --- Part B: decode.py — send_metadata failure must not kill the scheduler ---
dec = Path("/sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py")
text = dec.read_text()
B_OLD = (
    "            page_indices = kv_to_page_indices(kv_indices, kv_transfer_page_size)" + NL +
    "            decode_req.kv_receiver.send_metadata(" + NL +
    "                page_indices," + NL +
    "                decode_req.metadata_buffer_index," + NL +
    "                state_indices," + NL +
    "                decode_prefix_len=total_prefix_len," + NL +
    "            )"
)
B_NEW = (
    "            page_indices = kv_to_page_indices(kv_indices, kv_transfer_page_size)" + NL +
    "            try:" + NL +
    "                decode_req.kv_receiver.send_metadata(" + NL +
    "                    page_indices," + NL +
    "                    decode_req.metadata_buffer_index," + NL +
    "                    state_indices," + NL +
    "                    decode_prefix_len=total_prefix_len," + NL +
    "                )" + NL +
    "            except Exception as send_err:" + NL +
    "                # Peer bootstrap server busy/down (e.g. prefill restart)." + NL +
    "                # Sends are bounded by SNDTIMEO; free the slot and keep the" + NL +
    "                # request queued for retry instead of wedging the scheduler." + NL +
    "                self.req_to_metadata_buffer_idx_allocator.free(" + NL +
    "                    decode_req.metadata_buffer_index" + NL +
    "                )" + NL +
    "                decode_req.metadata_buffer_index = None" + NL +
    "                logger.warning(" + NL +
    "                    f\"send_metadata failed for room \"" + NL +
    "                    f\"{decode_req.req.bootstrap_room}, will retry: {send_err}\"" + NL +
    "                )" + NL +
    "                continue"
)
b_old, b_new = text.count(B_OLD), text.count(B_NEW)
if b_old == 1 and b_new == 0:
    text = text.replace(B_OLD, B_NEW, 1)
    applied.append("send_metadata_guard")
elif b_old == 0 and b_new == 1:
    pass
elif b_old == 0 and b_new == 0:
    print("PD_SEND_TIMEOUT: decode send_metadata anchor not found, skip")
else:
    raise RuntimeError(f"decode send_metadata anchor mismatch: old={b_old} new={b_new}")
dec.write_text(text)

# --- Part C: mooncake/conn.py — sync_status must not crash sender on dead peer ---
mc = Path("/sgl-workspace/sglang/python/sglang/srt/disaggregation/mooncake/conn.py")
text = mc.read_text()
C_OLD = (
    "        na = NetworkAddress(remote, dst_port)" + NL +
    "        self._connect(na.to_tcp(), is_ipv6=na.is_ipv6).send_multipart(" + NL +
    "            [" + NL +
    "                str(room).encode(\"ascii\")," + NL +
    "                str(status).encode(\"ascii\")," + NL +
    "                str(prefill_rank).encode(\"ascii\")," + NL +
    "            ]" + NL +
    "        )"
)
C_NEW = (
    "        na = NetworkAddress(remote, dst_port)" + NL +
    "        try:" + NL +
    "            self._connect(na.to_tcp(), is_ipv6=na.is_ipv6).send_multipart(" + NL +
    "                [" + NL +
    "                    str(room).encode(\"ascii\")," + NL +
    "                    str(status).encode(\"ascii\")," + NL +
    "                    str(prefill_rank).encode(\"ascii\")," + NL +
    "                ]" + NL +
    "            )" + NL +
    "        except Exception as e:" + NL +
    "            logger.warning(" + NL +
    "                f\"Failed to sync status for room {room} to {remote}:{dst_port}: {e}\"" + NL +
    "            )"
)
c_old, c_new = text.count(C_OLD), text.count(C_NEW)
if c_old == 1 and c_new == 0:
    text = text.replace(C_OLD, C_NEW, 1)
    applied.append("sync_status_guard")
elif c_old == 0 and c_new == 1:
    pass
elif c_old == 0 and c_new == 0:
    print("PD_SEND_TIMEOUT: mooncake sync_status already guarded or differs, skip")
else:
    raise RuntimeError(f"mooncake sync_status anchor mismatch: old={c_old} new={c_new}")
mc.write_text(text)

for p in (common, dec, mc):
    compile(p.read_text(), str(p), "exec")
print("PD_SEND_TIMEOUT_PATCH=" + ("applied:" + ",".join(applied) if applied else "already-patched"))
