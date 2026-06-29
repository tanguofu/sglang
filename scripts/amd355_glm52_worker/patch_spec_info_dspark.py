#!/usr/bin/env python3
"""Add DSPARK algorithm to spec_info.py (minimal patch for v0.5.14)."""
import os, re

FILE = "/sgl-workspace/sglang/python/sglang/srt/speculative/spec_info.py"
with open(FILE) as f:
    content = f.read()

if "DSPARK" in content:
    print("[PATCH] Already has DSPARK")
    exit(0)

# Add DSPARK enum value after DFLASH
content = content.replace(
    "    DFLASH = auto()",
    "    DFLASH = auto()\n    DSPARK = auto()"
)

# Add is_dspark method after is_dflash
content = content.replace(
    "    def is_dflash(self) -> bool:\n        return self == SpeculativeAlgorithm.DFLASH",
    "    def is_dflash(self) -> bool:\n        return self in (SpeculativeAlgorithm.DFLASH, SpeculativeAlgorithm.DSPARK)\n\n    def is_dspark(self) -> bool:\n        return self == SpeculativeAlgorithm.DSPARK"
)

# Add DSpark worker import in create_worker
content = content.replace(
    "        if self.is_dflash():\n            # V2 worker drives both overlap and non-overlap (scheduler runs it\n            # synchronously when overlap is disabled), same as EAGLE.\n            from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2\n\n            return DFlashWorkerV2",
    "        if self.is_dflash():\n            # V2 worker drives both overlap and non-overlap (scheduler runs it\n            # synchronously when overlap is disabled), same as EAGLE.\n            if self.is_dspark():\n                from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2\n\n                return DSparkWorkerV2\n            from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2\n\n            return DFlashWorkerV2"
)

with open(FILE, "w") as f:
    f.write(content)

print("[PATCH] Added DSPARK to spec_info.py")

import py_compile
py_compile.compile(FILE, doraise=True)
print("[VERIFY] Syntax OK")
