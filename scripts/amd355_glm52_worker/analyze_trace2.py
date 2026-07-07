import json, gzip, collections, glob

files = sorted(glob.glob('/data/profiles/dec3-*.trace.json.gz'))
trace_file = files[0]
with gzip.open(trace_file, 'rt') as f:
    data = json.load(f)
events = data['traceEvents']

kernel_times = collections.defaultdict(lambda: {'total': 0, 'count': 0})
for ev in events:
    cat = ev.get('cat', '')
    name = ev.get('name', '')
    dur = ev.get('dur', 0)
    if cat == 'kernel' and dur > 0:
        kernel_times[name]['total'] += dur
        kernel_times[name]['count'] += 1

total = sum(v['total'] for v in kernel_times.values())

# Group kernels by category
groups = {
    'MoE GEMM (fmoe)': [],
    'Communication (reduce/allgather)': [],
    'CK GEMM (blockscale)': [],
    'Quantization': [],
    'RMSNorm': [],
    'Attention/DSA': [],
    'Other GEMM': [],
    'Elementwise/Copy': [],
    'Other': [],
}

for name, info in kernel_times.items():
    nl = name.lower()
    if 'fmoe' in nl or 'grouped_topk' in nl or 'moe_sorting' in nl or 'append_shared' in nl:
        groups['MoE GEMM (fmoe)'].append((name, info))
    elif 'reduce_scatter' in nl or 'allreduce' in nl or 'allgather' in nl or 'cross_device' in nl:
        groups['Communication (reduce/allgather)'].append((name, info))
    elif 'blockscale' in nl or 'cshuffle' in nl:
        groups['CK GEMM (blockscale)'].append((name, info))
    elif 'quant' in nl:
        groups['Quantization'].append((name, info))
    elif 'rmsnorm' in nl or 'rms' in nl:
        groups['RMSNorm'].append((name, info))
    elif 'attn' in nl or 'attention' in nl or 'flash' in nl or 'tilelang' in nl or 'dsa' in nl or 'topk_transform' in nl or 'k_indexer' in nl or 'qk_rope' in nl:
        groups['Attention/DSA'].append((name, info))
    elif 'gemm' in nl or 'matmul' in nl or 'cijk' in nl:
        groups['Other GEMM'].append((name, info))
    elif 'elementwise' in nl or 'copy' in nl or 'fill' in nl or 'cat' in nl:
        groups['Elementwise/Copy'].append((name, info))
    else:
        groups['Other'].append((name, info))

print('=== Kernel time by category (total: %.1f ms) ===' % (total/1000))
for group_name, items in groups.items():
    if not items:
        continue
    group_total = sum(info['total'] for _, info in items)
    group_count = sum(info['count'] for _, info in items)
    pct = group_total / total * 100
    print('  %5.1f%%  %6.2f ms  (%d calls)  %s' % (pct, group_total/1000, group_count, group_name))
    for name, info in sorted(items, key=lambda x: -x[1]['total'])[:5]:
        print('         %.2f ms  (%d calls)  %s' % (info['total']/1000, info['count'], name[:90]))

# Also check for attention-related kernels specifically
print()
print('=== All kernels with attn/attention/flash/tilelang/dsa in name ===')
for name, info in sorted(kernel_times.items(), key=lambda x: -x[1]['total']):
    nl = name.lower()
    if any(k in nl for k in ['attn', 'attention', 'flash', 'tilelang', 'dsa', 'decode', 'prefill', 'sparse', 'indexer']):
        print('  %.2f ms  (%d calls)  %s' % (info['total']/1000, info['count'], name[:100]))
