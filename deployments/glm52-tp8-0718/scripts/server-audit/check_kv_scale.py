import json
import glob
import struct

# Read the safetensors index to look for kv_scale keys
try:
    with open("/data/model/glm52-fp8/model.safetensors.index.json") as f:
        idx = json.load(f)
    keys = list(idx.get("metadata", {}).keys()) + list(idx.get("weight_map", {}).keys())
    kv_scale_keys = [k for k in keys if "kv_scale" in k.lower() or "k_scale" in k.lower() or "v_scale" in k.lower()]
    print(f"Total keys in index: {len(keys)}")
    print(f"KV scale keys found: {len(kv_scale_keys)}")
    for k in kv_scale_keys[:20]:
        print(f"  {k}")
except FileNotFoundError:
    print("No index file - checking individual safetensors")
    files = sorted(glob.glob("/data/model/glm52-fp8/model-*.safetensors"))[:1]
    print(f"Checking first file: {files[0] if files else None}")
    if files:
        with open(files[0], "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
        kv_scale_keys = [k for k in header.keys() if "kv_scale" in k.lower() or "k_scale" in k.lower() or "v_scale" in k.lower()]
        print(f"KV scale keys in first shard: {len(kv_scale_keys)}")
        for k in kv_scale_keys[:20]:
            dtype = header[k].get("dtype", "?")
            print(f"  {k}: {dtype}")
