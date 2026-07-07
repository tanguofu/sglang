import json, gzip, glob, collections

files = sorted(glob.glob('/data/profiles/dec3-*.trace.json.gz'))
with gzip.open(files[0], 'rt') as f:
    data = json.load(f)
events = data['traceEvents']

# Sort kernel events by timestamp
kernel_events = [ev for ev in events if ev.get('cat') == 'kernel' and ev.get('dur', 0) > 0]
kernel_events.sort(key=lambda x: x.get('ts', 0))

# Find step boundaries (gaps > 0.5ms)
prev_end = 0
steps = []
current_step = []
for ev in kernel_events:
    ts = ev.get('ts', 0)
    if prev_end > 0 and ts - prev_end > 500:  # 0.5ms gap
        steps.append(current_step)
        current_step = []
    current_step.append(ev)
    prev_end = ts + ev.get('dur', 0)
if current_step:
    steps.append(current_step)

print('Detected %d step(s)' % len(steps))
for i, step in enumerate(steps):
    step_total = sum(ev.get('dur', 0) for ev in step)
    step_wall = (step[-1].get('ts',0) + step[-1].get('dur',0)) - step[0].get('ts',0)
    # Count unique kernel names
    names = collections.Counter(ev.get('name', '')[:60] for ev in step)
    top3 = names.most_common(3)
    print('  Step %d: %d kernels, kernel=%.1fms, wall=%.1fms, top: %s' % (i, len(step), step_total/1000, step_wall/1000, str(top3)[:150]))

# For the decode steps (smaller ones), show detailed breakdown
print()
for i, step in enumerate(steps):
    if len(step) < 300:  # decode steps
        step_total = sum(ev.get('dur', 0) for ev in step)
        if step_total < 5000:  # skip tiny steps
            print('=== Step %d detailed (kernel time: %.1f ms) ===' % (i, step_total/1000))
            ktimes = collections.defaultdict(lambda: {'total': 0, 'count': 0})
            for ev in step:
                name = ev.get('name', '')[:80]
                ktimes[name]['total'] += ev.get('dur', 0)
                ktimes[name]['count'] += 1
            for name, info in sorted(ktimes.items(), key=lambda x: -x[1]['total'])[:15]:
                print('  %.2f ms  (%d calls)  %s' % (info['total']/1000, info['count'], name))
            print()
