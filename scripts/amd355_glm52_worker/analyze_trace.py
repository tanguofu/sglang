import json, gzip, collections, sys, glob, os

files = sorted(glob.glob('/data/profiles/dec3-*.trace.json.gz'))
if not files:
    print('No trace files found!')
    sys.exit(1)

trace_file = files[0]
print('Loading:', trace_file, flush=True)
with gzip.open(trace_file, 'rt') as f:
    data = json.load(f)
events = data['traceEvents']
print('Total events:', len(events), flush=True)

kernel_times = collections.defaultdict(lambda: {'total': 0, 'count': 0})
cat_times = collections.defaultdict(lambda: {'total': 0, 'count': 0})

for ev in events:
    cat = ev.get('cat', '')
    name = ev.get('name', '')
    dur = ev.get('dur', 0)
    if cat == 'kernel' and dur > 0:
        key = name[:120] if len(name) > 120 else name
        kernel_times[key]['total'] += dur
        kernel_times[key]['count'] += 1
    if dur > 0:
        cat_times[cat]['total'] += dur
        cat_times[cat]['count'] += 1

print()
print('=== Category breakdown ===', flush=True)
for cat, info in sorted(cat_times.items(), key=lambda x: -x[1]['total']):
    print('  %s: total=%.1f ms, count=%d' % (cat, info['total']/1000, info['count']), flush=True)

total_kernel = sum(v['total'] for v in kernel_times.values())
print()
print('=== Total kernel time: %.1f ms ===' % (total_kernel/1000), flush=True)
print()
print('=== Top 30 GPU kernels by total time ===', flush=True)
for name, info in sorted(kernel_times.items(), key=lambda x: -x[1]['total'])[:30]:
    pct = info['total'] / total_kernel * 100 if total_kernel > 0 else 0
    print('  %5.1f%%  %.2f ms  (%d calls)  avg=%.3f ms  %s' % (pct, info['total']/1000, info['count'], info['total']/info['count']/1000, name[:100]), flush=True)
