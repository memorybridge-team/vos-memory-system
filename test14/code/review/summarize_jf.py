#!/usr/bin/env python3
"""J&F by window with video-clustered paired bootstrap against a reference method."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

tag = sys.argv[1]
reference = sys.argv[2] if len(sys.argv) > 2 else 'target_ref'
ROOT = '/workspace/cmmt-design-review-20261104/runs/jf/'
rows = [json.load(open(f)) for f in glob.glob(ROOT + tag + '/cases/*.json')]
print(tag, 'cases', len(rows), 'videos', len({r['case']['video_id'] for r in rows}))
methods = list(rows[0]['scores'])
WINDOWS = {'+1..10': (1, 10), '+11..30': (11, 30), '+31..100': (31, 100), '+101..': (101, 10 ** 9), 'all': (1, 10 ** 9)}


def window_value(row, method, low, high):
    s = row['scores'][method]
    picked = [(j + f) / 2 for o, j, f, v in zip(row['offsets'], s['J'], s['F'], s['visible']) if low <= o <= high and j is not None and v]
    return float(np.mean(picked)) if picked else None


def per_video(method, low, high):
    grouped = defaultdict(list)
    for row in rows:
        value = window_value(row, method, low, high)
        if value is not None:
            grouped[row['case']['video_id']].append(value)
    return {k: float(np.mean(v)) for k, v in grouped.items()}


rng = np.random.default_rng(7)
for name, (low, high) in WINDOWS.items():
    print(f'\n[{name}] J&F (visible frames, video mean, x100)')
    base = per_video(reference, low, high)
    for method in methods:
        value = per_video(method, low, high)
        keys = sorted(set(base) & set(value))
        if not keys:
            continue
        a, b = np.array([value[k] for k in keys]), np.array([base[k] for k in keys])
        diff = a - b
        samples = diff[rng.integers(0, len(diff), (5000, len(diff)))].mean(1)
        low_ci, high_ci = np.percentile(samples, [2.5, 97.5])
        worst = np.sort(diff)[:3]
        print(f'  {method:24s} {a.mean() * 100:7.2f}  diff {diff.mean() * 100:+6.2f} [{low_ci * 100:+.2f},{high_ci * 100:+.2f}]  n={len(keys)}  worst3 {np.round(worst * 100, 1).tolist()}')
print('\nfailures: videos more than 10 points below / above the reference (all frames)')
base = per_video(reference, 1, 10 ** 9)
for method in methods:
    value = per_video(method, 1, 10 ** 9)
    keys = sorted(set(base) & set(value))
    diff = np.array([value[k] - base[k] for k in keys])
    lost = [k for k in keys if value[k] - base[k] < -0.10]
    print(f'  {method:24s} below {int((diff < -0.10).sum()):3d}  above {int((diff > 0.10).sum()):3d}  of {len(keys)}  {lost[:5]}')
print('\nnormalized state error (case mean):')
for method in methods:
    e = [r['errors'][method] for r in rows]
    print(f"  {method:24s} spatial {np.mean([x['spatial'] for x in e]):.4f} pointer {np.mean([x['pointer'] for x in e]):.4f}")
print('seconds per case', round(float(np.mean([r['seconds'] for r in rows])), 1))
