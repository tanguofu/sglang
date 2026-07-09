from pathlib import Path
p = Path('/sgl-workspace/sglang/python/sglang/jit_kernel/fused_store_index_cache.py')
s = p.read_text()
start = s.index('def can_use_dsa_fused_store(')
end = s.index('\n\n@debug_kernel_api', start)
replacement = '''def can_use_dsa_fused_store(
    key_dtype, indices_dtype, page_size
) -> bool:
    return False
'''
p.write_text(s[:start] + replacement + s[end:])
print('[FIX] DSA fused store disabled: can_use_dsa_fused_store -> False')
