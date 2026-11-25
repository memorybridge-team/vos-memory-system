#!/usr/bin/env python3
"""Live handoff results: video-clustered means, paired differences and failure counts."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

tag = sys.argv[1]
ROOT = '/workspace/cmmt-design-review-20261104/runs/unseen/'
cases = [json.load(open(f)) for f in glob.glob(ROOT + tag + '/cases/*.json')]
rows = []
for case in cases:
    for name, row in case['switches'].items():
        rows.append(dict(video=case['video'], obj=case['object'], switch=name, row=row))
print(tag, 'objects', len(cases), 'switch cases', len(rows), 'videos', len({r['video'] for r in rows}))
methods = [k for k in rows[0]['row'] if isinstance(rows[0]['row'][k], dict) and k != 'trace' and not k.endswith('_first30')]


def per_video(method, key):
    grouped = defaultdict(list)
    for r in rows:
        value = r['row'].get(method, {}).get(key)
        if value is not None:
            grouped[r['video']].append(value)
    return {k: float(np.mean(v)) for k, v in grouped.items()}


rng = np.random.default_rng(7)
for label, suffix, key in (('all frames after switch: J', '', 'J'), ('all frames after switch: J&F', '', 'JF'), ('first 30 frames: J&F', '_first30', 'JF')):
    print(f'\n[{label}] video mean x100, difference from Base+ Full Replay')
    base = per_video('base' + suffix, key)
    for method in methods:
        value = per_video(method + suffix, key)
        keys = sorted(set(base) & set(value))
        if not keys:
            continue
        a, b = np.array([value[k] for k in keys]), np.array([base[k] for k in keys])
        diff = a - b
        samples = diff[rng.integers(0, len(diff), (5000, len(diff)))].mean(1)
        low, high = np.percentile(samples, [2.5, 97.5])
        print(f'  {method:18s} {a.mean() * 100:7.2f}  diff {diff.mean() * 100:+6.2f} [{low * 100:+.2f},{high * 100:+.2f}]  n={len(keys)}')
print('\nfailures: switch cases with J at least 0.2 below Base+ while Base+ J >= 0.5')
good = [r for r in rows if (r['row']['base'].get('J') or 0) >= 0.5]
print('  eligible cases', len(good))
for method in methods:
    lost = [r for r in good if r['row'].get(method, {}).get('J') is not None and r['row'][method]['J'] < r['row']['base']['J'] - 0.2]
    print(f"  {method:18s} {len(lost):4d}  {[r['video'] + ':' + r['switch'] for r in lost[:4]]}")
