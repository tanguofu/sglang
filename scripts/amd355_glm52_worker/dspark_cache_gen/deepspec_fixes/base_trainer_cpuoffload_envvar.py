# =============================================================================
# PATCH: Make CPUOffload configurable via DS_CPU_OFFLOAD env var
# =============================================================================
# Problem: base_trainer.py hardcodes CPUOffload(offload_params=True) in
#   _build_fsdp_kwargs(). This was added for 1-node OOM mitigation but makes
#   4-node training extremely slow (~60h) because params are constantly
#   moved between CPU and GPU.
#
# Fix: Check DS_CPU_OFFLOAD env var. Default to "0" (off) for multi-node.
#   - DS_CPU_OFFLOAD=1 → CPUOffload(offload_params=True)  (1-node OOM fallback)
#   - DS_CPU_OFFLOAD=0 → CPUOffload(offload_params=False) (4-node, no offload)
#
# Apply: Replace the cpu_offload line in _build_fsdp_kwargs() in
#   /data/DeepSpec/deepspec/trainer/base_trainer.py
#
# Original line (line 68):
#       cpu_offload=CPUOffload(offload_params=True),
#
# Replacement:
#       cpu_offload=CPUOffload(
#           offload_params=os.environ.get("DS_CPU_OFFLOAD", "0") == "1"
#       ),
# =============================================================================

# The full patched _build_fsdp_kwargs function:

PATCHED_FUNCTION = '''
def _build_fsdp_kwargs(
    *, sharding_strategy_name: str, precision_dtype, world_size: int
) -> dict:
    sharding_strategy = _SHARDING_STRATEGIES[sharding_strategy_name]
    # CPUOffload controlled by DS_CPU_OFFLOAD env var:
    #   "1" = on  (1-node OOM fallback, very slow)
    #   "0" = off (4-node, params fit in VRAM across 32 GPUs)
    _offload_params = os.environ.get("DS_CPU_OFFLOAD", "0") == "1"
    fsdp_kwargs = dict(
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=precision_dtype,
            buffer_dtype=precision_dtype,
        ),
        sharding_strategy=sharding_strategy,
        cpu_offload=CPUOffload(offload_params=_offload_params),
    )
    if sharding_strategy in _HYBRID_STRATEGIES:
        devices_per_node = torch.cuda.device_count()
        fsdp_kwargs["device_mesh"] = init_device_mesh(
            "cuda",
            (world_size // devices_per_node, devices_per_node),
            mesh_dim_names=("replicate", "shard"),
        )
    return fsdp_kwargs
'''

# Shell command to apply the patch on each node:
APPLY_COMMAND = r'''
python3 -c "
import re
path = '/data/DeepSpec/deepspec/trainer/base_trainer.py'
with open(path) as f:
    content = f.read()

# Check if already patched
if 'DS_CPU_OFFLOAD' in content:
    print('Already patched')
else:
    # Replace the hardcoded cpu_offload line
    old = 'cpu_offload=CPUOffload(offload_params=True),'
    new = 'cpu_offload=CPUOffload(offload_params=os.environ.get(\"DS_CPU_OFFLOAD\", \"0\") == \"1\"),'
    if old in content:
        content = content.replace(old, new)
        with open(path, 'w') as f:
            f.write(content)
        print('Patched successfully')
    else:
        print('WARNING: Could not find target line to patch')
"
'''
