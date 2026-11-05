#!/usr/bin/env python3
"""Details behind paired_acc: is the earlier legacy run the same as t2, all-case gain results, and what a refresh does.

Usage: acc_detail.py <mose|val>
"""
import glob
import json
import os
import sys

import numpy as np

JF = '/workspace/cmmt-design-review-20261104/runs/jf/'


def score(row, method, low=1, high=10 ** 9):
    s = row['scores'][method]
    picked = [(j + f) / 2 for o, j, f, v in zip(row['offsets'], s['J'], s['F'], s['visible'])
              if low <= o <= high and j is not None and v]
    return float(np.mean(picked)) if picked else None


def boot(diff, seed=7):
    rng = np.random.default_rng(seed)
    s = diff[rng.integers(0, len(diff), (20000, len(diff)))].mean(1)
    return tuple(np.percentile(s, [2.5, 97.5]) * 100)


def load(tag):
    return {os.path.basename(p): json.load(open(p)) for p in glob.glob(f'{JF}{tag}/cases/*.json')}


def line(name, diff):
    diff = np.array(diff)
    ci = boot(diff)
    print(f'  {name:34s} {diff.mean() * 100:+6.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] n={len(diff)} win {int((diff > .10).sum())} lose {int((diff < -.10).sum())}')


key = sys.argv[1]
sens, t2, t5 = load(f'sens_{key}'), load(f't2_{key}'), load(f't5_{key}')
print('sens config', json.load(open(f'{JF}sens_{key}/config.json')).get('autocast'), 't2 config', json.load(open(f'{JF}t2_{key}/config.json')).get('autocast'))
print('sens methods', sorted(next(iter(sens.values()))['scores']))
for name in ('legacy', 'target_ref', 'target'):
    shared = [f for f in sens if f in t2 and name in sens[f]['scores'] and name in t2[f]['scores']]
    if shared:
        gaps = [abs((score(sens[f], name) or 0) - (score(t2[f], name) or 0)) for f in shared]
        print(f'{name}: {len(shared)} shared cases, max gap {max(gaps) * 100:.3f}, mean gap {np.mean(gaps) * 100:.3f}')

# Reference rows for every case: t5 target_ref, and legacy from sens when it matches t2.
if t5:
    names = [m for m in next(iter(t5.values()))['scores'] if m != 'target_ref']
    print('\nagainst legacy from the earlier run, all cases')
    for m in names:
        pairs = [(score(t5[f], m), score(sens[f], 'legacy')) for f in t5 if f in sens and 'legacy' in sens[f]['scores']]
        diff = [a - b for a, b in pairs if a is not None and b is not None]
        if diff:
            line(f'{m} - legacy', diff)
    base = [score(sens[f], 'legacy') for f in t5 if f in sens and 'legacy' in sens[f]['scores']]
    base = [b for b in base if b is not None]
    print(f'  legacy mean {np.mean(base) * 100:.2f} n={len(base)}')

    print('\nrefresh detail')
    for m in names:
        offsets = [t5[f]['refreshed'].get(m) for f in t5]
        if not any(o is not None for o in offsets):
            continue
        done = [o for o in offsets if o is not None]
        print(f' {m}: refreshed in {len(done)}/{len(offsets)} cases, median offset {np.median(done):.0f}, p90 {np.percentile(done, 90):.0f}')
        source = 'legacy'
        diff, harm = [], []
        for f in t5:
            o = t5[f]['refreshed'].get(m)
            if o is None or f not in sens or 'legacy' not in sens[f]['scores']:
                continue
            a, b = score(t5[f], m, o + 1), score(sens[f], source, o + 1)
            if a is not None and b is not None:
                diff.append(a - b)
        if len(diff) >= 8:
            line('after the refresh, vs legacy', diff)
