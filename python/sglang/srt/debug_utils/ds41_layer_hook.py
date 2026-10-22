"""Forward-hook probes for the DeepSeek-V4.1 decode investigation.

Attached through ``--forward-hooks``, which takes JSON like::

    [{"name": "mlp",
      "target_modules": ["model.layers.*.mlp"],
      "hook_factory": "sglang.srt.debug_utils.ds41_layer_hook:make",
      "config": {"path": "/tmp/hook.tsv"}}]

``make(config)`` returns the hook itself.  Each call appends one row:

    rank  layer  call_index  abs_sum  square_sum  max_abs

``call_index`` is a per-process counter that increments on every hooked call,
so ``call_index // layers_per_forward`` is the forward pass and the remainder is
the layer, which lets rows from different requests be aligned exactly instead of
by position.

Nothing runs unless the config names a file.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

_COUNTER = {"n": 0}


def make(config: Dict[str, Any]) -> Optional[Any]:
    path = config.get("path") or os.environ.get("SGLANG_DS41_HOOK_PATH")
    if not path:
        return None
    tag = str(config.get("layer", config.get("tag", "")))
    counter = _COUNTER

    def hook(module, inputs, output):
        try:
            tensor = output
            if isinstance(tensor, (tuple, list)):
                tensor = next((t for t in tensor if hasattr(t, "detach")), None)
            if tensor is None or not hasattr(tensor, "detach"):
                return
            counter["n"] += 1
            value = tensor.detach().float()
            with open(path, "a") as probe_file:
                probe_file.write(
                    "%s\t%s\t%d\t%.6f\t%.6f\t%.6f\n"
                    % (
                        os.environ.get("RANK", "-1"),
                        tag,
                        counter["n"],
                        float(value.abs().sum()),
                        float(value.pow(2).sum()),
                        float(value.abs().max()),
                    )
                )
        except Exception:  # a probe must never disturb a forward pass
            pass

    return hook
