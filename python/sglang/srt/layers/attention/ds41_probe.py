"""File-driven debug probes for the DeepSeek-V4.1 gfx942 investigation.

A restart costs a full JIT warmup, so probe settings live in a JSON file that is
re-read whenever it changes: settling one question at a time should not require
rebuilding and redeploying.  Everything here is off unless the control file
exists, and every failure is swallowed so a probe can never disturb a forward.
"""

import json
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

CONTROL_PATH = os.environ.get("SGLANG_DS41_PROBE_CFG", "/probe/cfg")

# Master switch read once: with it off every probe costs one attribute lookup,
# so the calls can stay at their call sites instead of being conditionally
# sprinkled through the hot path.
ENABLED = os.environ.get("SGLANG_DS41_PROBE") == "1"

_state = {"mtime": None, "config": {}}

_current_layer = [-1]


def set_layer(layer_id: int) -> None:
    """Kernel-level probes have no layer argument; the backend stamps it here."""
    _current_layer[0] = int(layer_id)


def current_layer() -> int:
    return _current_layer[0]


def config() -> dict:
    try:
        mtime = os.stat(CONTROL_PATH).st_mtime
    except OSError:
        _state["config"] = {}
        _state["mtime"] = None
        return _state["config"]
    if _state["mtime"] != mtime:
        try:
            with open(CONTROL_PATH) as control_file:
                loaded = json.load(control_file)
            _state["config"] = loaded if isinstance(loaded, dict) else {}
        except Exception:
            _state["config"] = {}
        _state["mtime"] = mtime
    return _state["config"]


def value(name: str, default=None):
    settings = config()
    if name in settings:
        return settings[name]
    return os.environ.get(name.upper(), default)


def flag(name: str) -> bool:
    return value(name) not in (None, False, 0, "", "0")


def layers(name: str = "layers", default: str = "1"):
    raw = value(name, default)
    if isinstance(raw, (list, tuple)):
        return frozenset(int(x) for x in raw)
    try:
        return frozenset(int(x) for x in str(raw).split(",") if x.strip())
    except ValueError:
        return frozenset({1})


def write(path: str, line: str) -> None:
    """Append one row to a per-process probe file."""
    with open("%s.%d" % (path, os.getpid()), "a") as probe_file:
        probe_file.write(line)


def _gather_slots(cache: torch.Tensor, slots: torch.Tensor, length: int):
    """The bytes a row's attention kernel reads, in slot order."""
    page_size = cache.shape[1]
    slots = slots[:length]
    keep = (slots >= 0) & (slots < cache.shape[0] * page_size)
    slots = slots[keep]
    if slots.numel() == 0:
        return None, int((~keep).sum().item())
    page = (slots // page_size).to(torch.int64)
    row = (slots % page_size).to(torch.int64)
    gathered = cache[page, row]
    return gathered.reshape(gathered.shape[0], -1).view(torch.uint8), int(
        (~keep).sum().item()
    )


def _slot_preview(slots: torch.Tensor, length: int, count: int = 4) -> str:
    head = slots[: min(count, length)].tolist()
    tail = slots[max(0, length - count) : length].tolist()
    return "%s|%s" % (head, tail)


def probe_kv_read(
    layer_id: int,
    compress_ratio: int,
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_page_indices: torch.Tensor,
    swa_topk_lengths,
    extra_k_cache,
    extra_indices,
    extra_topk_lengths,
) -> None:
    """Compare what the two twin rows of one forward read out of the KV cache.

    Both rows carry the same prompt, so their attention must gather identical
    bytes.  ``swa`` / ``extra`` report the largest per-slot byte difference;
    ``ninv*`` counts -1 holes inside the window, which would silently shorten
    one row's context even when every byte it does read is correct.
    """
    path = value("kvread")
    if not ENABLED or not path or layer_id not in layers("kvread_layers", "1"):
        return
    if torch.cuda.is_current_stream_capturing():
        # A synchronize inside capture invalidates the whole graph.
        return
    try:
        if swa_page_indices.dim() == 3:
            swa_page_indices = swa_page_indices[:, 0]
        if extra_indices is not None and extra_indices.dim() == 3:
            extra_indices = extra_indices[:, 0]
        rows = swa_page_indices.shape[0]
        if rows < 2:
            return
        torch.cuda.synchronize()

        q2 = q.reshape(rows, -1).detach().float()
        q_diff = float((q2[0] - q2[1]).abs().max())

        def length_of(lengths, index):
            if lengths is None:
                return int(swa_page_indices.shape[-1])
            return min(int(lengths.reshape(-1)[index].item()), swa_page_indices.shape[-1])

        len0, len1 = length_of(swa_topk_lengths, 0), length_of(swa_topk_lengths, 1)
        kv0, ninv0 = _gather_slots(swa_k_cache, swa_page_indices[0], len0)
        kv1, ninv1 = _gather_slots(swa_k_cache, swa_page_indices[1], len1)
        swa_diff, swa_at, bad_slots = -1.0, -1, ""
        if kv0 is not None and kv1 is not None and kv0.shape == kv1.shape:
            delta = (kv0.float() - kv1.float()).abs()
            per_slot = delta.max(dim=1).values
            swa_at = int(per_slot.argmax().item())
            swa_diff = float(per_slot.max().item())
            bad = (per_slot > 0).nonzero().flatten()
            bad_slots = ",".join(
                "%d:%d/%d"
                % (
                    int(i),
                    int(swa_page_indices[0].reshape(-1)[i].item()),
                    int(swa_page_indices[1].reshape(-1)[i].item()),
                )
                for i in bad[:6].tolist()
            )

        extra_diff, extra_at, extra_n = -1.0, -1, 0
        if extra_k_cache is not None and extra_indices is not None:
            e0, _ = _gather_slots(extra_k_cache, extra_indices[0], length_of(extra_topk_lengths, 0))
            e1, _ = _gather_slots(extra_k_cache, extra_indices[1], length_of(extra_topk_lengths, 1))
            if e0 is not None and e1 is not None and e0.shape == e1.shape:
                delta = (e0.float() - e1.float()).abs()
                per_slot = delta.max(dim=1).values
                extra_at = int(per_slot.argmax().item())
                extra_diff = float(per_slot.max().item())
                extra_n = int((per_slot > 0).sum().item())

        rank = int(os.environ.get("RANK", torch.cuda.current_device()))
        write(
            path,
            "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.6f\t%.8f\t%d\t%d\t%.8f\t%d\t%s\t%s\t%s\n"
            % (
                rank,
                layer_id,
                int(compress_ratio),
                len0,
                len1,
                ninv0,
                ninv1,
                q_diff,
                swa_diff,
                swa_at,
                extra_n,
                extra_diff,
                extra_at,
                _slot_preview(swa_page_indices[0], len0),
                _slot_preview(swa_page_indices[1], len1),
                bad_slots,
            ),
        )
    except Exception:  # a probe must never disturb a forward pass
        logger.exception("kv-read probe failed")


_decode_calls = {}


def _dump_dir() -> str:
    return value("dump_dir", "/probe/dump")


def _dump_append(name: str, header: str, arrays) -> None:
    directory = _dump_dir()
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        return
    path = os.path.join(directory, "%s.%d" % (name, os.getpid()))
    with open(path, "ab") as dump_file:
        dump_file.write(header.encode())
        for array in arrays:
            dump_file.write(memoryview(np.ascontiguousarray(array)).cast("B"))


def probe_store(layer_id: int, swa_k: torch.Tensor, swa_loc: torch.Tensor) -> None:
    """Record where a decode step writes its K and what it writes.

    The read side already knows which slots the kernel uses; comparing the two
    says whether a store lands on the slot its own row will read.
    """
    if not ENABLED or not flag("store_probe") or layer_id not in layers("store_layers", "1"):
        return
    if torch.cuda.is_current_stream_capturing():
        return
    if int(swa_k.shape[0]) != 2:
        # warmup batches are larger; only the two-row twin request is interesting
        return
    try:
        if not value("dump_dir"):
            return
        torch.cuda.synchronize()
        rows = min(2, int(swa_loc.numel()))
        locs = swa_loc.reshape(-1)[:8].to(torch.int32).cpu().numpy()
        k = swa_k.detach().float()
        sums = [float(k[i].abs().sum()) for i in range(min(rows, k.shape[0]))]
        call = _decode_calls.get(("store", layer_id), 0)
        _decode_calls[("store", layer_id)] = call + 1
        if call >= int(value("dump_calls", 4)):
            return
        _dump_append(
            "store",
            "STORE rank=%s layer=%d call=%d locs=%s ksum=%s rows=%d\n"
            % (
                os.environ.get("RANK", torch.cuda.current_device()),
                layer_id,
                call,
                locs.tolist(),
                ["%.6f" % x for x in sums],
                int(swa_k.shape[0]),
            ),
            [swa_loc.reshape(-1)[:8].to(torch.int32).cpu().numpy()],
        )
    except Exception:
        logger.exception("store probe failed")


def probe_kv_dump(
    layer_id: int,
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_page_indices: torch.Tensor,
    swa_topk_lengths,
) -> None:
    """Dump the exact bytes the two rows gather, so the difference can be
    classified offline instead of guessed at from a max over uint8."""
    if not ENABLED or not flag("kvread_dump") or layer_id not in layers("dump_layers", "1"):
        return
    if torch.cuda.is_current_stream_capturing():
        return
    try:
        if not value("dump_dir"):
            return
        if swa_page_indices.dim() == 3:
            swa_page_indices = swa_page_indices[:, 0]
        if swa_page_indices.shape[0] != 2:
            return
        call = _decode_calls.get(("kv", layer_id), 0)
        _decode_calls[("kv", layer_id)] = call + 1
        if call >= int(value("dump_calls", 4)):
            return
        torch.cuda.synchronize()

        def length_of(index):
            if swa_topk_lengths is None:
                return int(swa_page_indices.shape[-1])
            return min(
                int(swa_topk_lengths.reshape(-1)[index].item()),
                int(swa_page_indices.shape[-1]),
            )

        parts = []
        head = "KV rank=%s layer=%d call=%d len0=%d len1=%d dim=%d\n" % (
            os.environ.get("RANK", torch.cuda.current_device()),
            layer_id,
            call,
            length_of(0),
            length_of(1),
            int(swa_k_cache.shape[-1]),
        )
        for index in (0, 1):
            length = length_of(index)
            slots = swa_page_indices[index][:length].to(torch.int32).cpu().numpy()
            page_size = swa_k_cache.shape[1]
            page = torch.from_numpy(slots.astype(np.int64)) // page_size
            row = torch.from_numpy(slots.astype(np.int64)) % page_size
            gathered = swa_k_cache[page, row].reshape(len(slots), -1)
            parts.append(slots)
            parts.append(gathered.view(torch.uint8).cpu().numpy())
        _dump_append("kv", head, parts)
    except Exception:
        logger.exception("kv dump probe failed")


def probe_store_kv(
    layer_id: int,
    kv: torch.Tensor,
    swa_loc: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Compare the bf16 KV the two twin rows hand to the fused store.

    The store kernel is one block per token with a per-tile scale, so identical
    inputs must give identical bytes.  A difference here is upstream of the
    store (the kv projection), not in the quantization.
    """
    if not ENABLED or not flag("store_kv_probe") or layer_id not in layers("store_kv_layers", "0,1"):
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    tokens = int(kv.shape[0])
    if tokens < 2 or tokens % 2:
        return
    half = tokens // 2
    try:
        torch.cuda.synchronize()
        rows = kv.detach().float()
        if half == 1:
            diff = float((rows[0] - rows[1]).abs().max())
        else:
            diff = float((rows[:half] - rows[half:]).abs().max())
        call = _decode_calls.get(("storekv", layer_id), 0)
        _decode_calls[("storekv", layer_id)] = call + 1
        head = (
            "STOREKV rank=%s layer=%d call=%d rows=%d dim=%d pos=%s loc=%s "
            "maxdiff=%.9f sum0=%.6f sum1=%.6f\n"
            % (
                os.environ.get("RANK", torch.cuda.current_device()),
                layer_id,
                call,
                tokens,
                int(kv.shape[-1]),
                positions.reshape(-1)[:4].to(torch.int32).cpu().numpy().tolist(),
                swa_loc.reshape(-1)[:4].to(torch.int32).cpu().numpy().tolist(),
                diff,
                float(rows[:half].abs().sum()),
                float(rows[half:].abs().sum()),
            )
        )
        if call < int(value("dump_calls", 4)) and value("dump_dir"):
            _dump_append(
                "storekv",
                head,
                [kv[:2].detach().view(torch.uint8).cpu().numpy()],
            )
        else:
            _dump_append("storekvlog", head, [])
    except Exception:
        logger.exception("store-kv probe failed")


def probe_store_roundtrip(
    layer_id: int, cache: torch.Tensor, swa_loc: torch.Tensor, page_size: int, dim: int
) -> None:
    """Read back every row the store just wrote and compare the twin rows.

    The fused store gets bit-identical input for both rows (probe_store_kv), so
    the bytes it leaves behind must be identical too.  Prefill writes the whole
    prompt in one launch, which is where a tile-shared scale or a misaligned
    row would show up; decode writes one token and is the control.
    """
    if not ENABLED or not flag("roundtrip_probe") or layer_id not in layers("roundtrip_layers", "0,1"):
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    total = int(swa_loc.numel())
    if total < 2 or total % 2:
        return
    half = total // 2
    try:
        torch.cuda.synchronize()
        slots = swa_loc.reshape(-1).to(torch.int64)
        page = (slots // page_size)
        row = (slots % page_size)
        gathered = cache[page, row].reshape(total, -1).view(torch.uint8).cpu().numpy()
        left, right = gathered[:half].astype(np.int16), gathered[half:].astype(np.int16)
        per_row = np.abs(left - right).max(axis=1)
        diff_rows = int((per_row > 0).sum())
        diff_bytes = int((left != right).sum())
        worst = int(per_row.argmax())
        call = _decode_calls.get(("roundtrip", layer_id), 0)
        _decode_calls[("roundtrip", layer_id)] = call + 1
        write(
            value("roundtrip_log", "/probe/roundtrip"),
            "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n"
            % (
                os.environ.get("RANK", torch.cuda.current_device()),
                layer_id,
                call,
                total,
                int(slots[0].item()),
                int(slots[half].item()),
                diff_rows,
                diff_bytes,
                worst,
                int(per_row[worst].item()),
            ),
        )
    except Exception:
        logger.exception("store roundtrip probe failed")


def probe_halves(tag: str, tensor) -> None:
    """Compare the two identical prompt halves of one tensor inside a forward.

    Gates on the control file so a single restart can triangulate the remaining
    stage list without another rebuild.
    """
    if not ENABLED or not flag("halves_probe"):
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    if not isinstance(tensor, torch.Tensor) or tensor.dim() < 2:
        return
    rows = int(tensor.shape[0])
    if rows < 4 or rows % 2:
        return
    try:
        torch.cuda.synchronize()
        flat = tensor.reshape(rows, -1).float()
        half = rows // 2
        left, right = flat[:half], flat[half:]
        write(
            value("halves_log", "/probe/halves"),
            "%s\t%d\t%d\t%.9f\t%.6f\t%.6f\n"
            % (
                os.environ.get("RANK", torch.cuda.current_device()),
                current_layer(),
                rows,
                float((left - right).abs().max()),
                float(left.abs().sum()),
                float(right.abs().sum()),
            ),
        )
    except Exception:
        logger.exception("halves probe failed")


def probe_q_halves(layer_id: int, q: torch.Tensor) -> None:
    """Compare the query projection of the two identical prompts in one call.

    ``attn_in`` is bit-identical at layer 0 and the attention output is not, so
    splitting q (identical projection input, identical weights) decides between
    the projection GEMM and the attention kernel.
    """
    if not ENABLED or not flag("q_probe") or layer_id not in layers("q_layers", "0,1,2"):
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    rows = int(q.shape[0])
    if rows < 4 or rows % 2:
        return
    try:
        torch.cuda.synchronize()
        half = rows // 2
        q_f = q.detach().float()
        left, right = q_f[:half], q_f[half:]
        write(
            value("q_log", "/probe/q"),
            "%s\t%d\t%d\t%.9f\t%.6f\t%.6f\t%.6f\t%.6f\n"
            % (
                os.environ.get("RANK", torch.cuda.current_device()),
                layer_id,
                rows,
                float((left - right).abs().max()),
                float(left.abs().sum()),
                float(right.abs().sum()),
                float(left.pow(2).sum()),
                float(right.pow(2).sum()),
            ),
        )
    except Exception:
        logger.exception("q probe failed")
