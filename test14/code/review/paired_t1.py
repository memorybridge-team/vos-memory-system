#!/usr/bin/env python3
"""Paired comparisons for the memory compositions: all frames, by window, and for hard groups."""
import glob
import json
import sys

import numpy as np

JF = '/workspace/cmmt-design-review-20261104/runs/jf/'
WINDOWS = {'all': (1, 10 ** 9), '+1..10': (1, 10), '+11..30': (11, 30), '+31..100': (31, 100), '+101..': (101, 10 ** 9)}
PAIRS = [
    ('target_ref', 'small'),      # gate B: is switching to Base+ worth anything here?
    ('olv', 'small'),
    ('ridge_nc', 'small'),
    ('ridge_nc_anchor', 'small'),
    ('olv', 'ridge_nc'),          # stop clause: does the state-free method match the translated one?
    ('olv', 'legacy_nc'),
    ('olv', 'ridge_nc_anchor'),   # does translation add anything on top of the two anchors?
    ('ridge_nc_anchor', 'ridge_nc'),
    ('ridge_nc', 'legacy_nc'),
    ('ridge_nc', 'orig'),         # do the translated recent records pay for themselves?
    ('ridge_nc', 'direct_nc'),
    ('ridge_nc', 'lastvis'),
    ('olv', 'orig'),
    ('olv', 'lastvis'),
    ('recent_ridge', 'lastvis'),
    ('ridge_nc_anchor', 'ridge_anchor'),
]


def boot(diff, seed=7):
    rng = np.random.default_rng(seed)
    s = diff[rng.integers(0, len(diff), (20000, len(diff)))].mean(1)
    return tuple(np.percentile(s, [2.5, 97.5]) * 100)


def value(row, method, low, high):
    s = row['scores'][method]
    picked = [(j + f) / 2 for o, j, f, v in zip(row['offsets'], s['J'], s['F'], s['visible'])
              if low <= o <= high and j is not None and v]
    return float(np.mean(picked)) if picked else None


def table(rows, title):
    print(f'\n== {title} ({len(rows)} cases)')
    for name, (low, high) in WINDOWS.items():
        print(f' [{name}]')
        for a, b in PAIRS:
            pairs = [(value(r, a, low, high), value(r, b, low, high)) for r in rows]
            diff = np.array([x - y for x, y in pairs if x is not None and y is not None])
            if len(diff) < 5:
                continue
            ci = boot(diff)
            print(f'  {a:16s} - {b:16s} {diff.mean() * 100:+6.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]  n={len(diff)}  win {int((diff > .10).sum())} lose {int((diff < -.10).sum())}')


def absent_gap(row):
    """Frames from the switch to the object's first visible frame in the target rollout."""
    for offset, visible in zip(row['offsets'], row['scores']['target_ref']['visible']):
        if visible:
            return offset
    return None


def split_groups(rows):
    groups = {}
    full = [(r, value(r, 'target_ref', 1, 10 ** 9), value(r, 'small', 1, 10 ** 9)) for r in rows]
    full = [(r, t, s) for r, t, s in full if t is not None and s is not None]
    base = np.array([t for _, t, _ in full])
    low, high = np.percentile(base, [33.3, 66.7])
    groups['hard: target_ref in lowest third'] = [r for r, t, _ in full if t <= low]
    groups['easy: target_ref in highest third'] = [r for r, t, _ in full if t >= high]
    groups['absent at switch (first visible > +1)'] = [r for r in rows if (absent_gap(r) or 0) > 1]
    groups['visible at switch'] = [r for r in rows if absent_gap(r) == 1]
    return groups


for tag in sys.argv[1:]:
    rows = [json.load(open(f)) for f in glob.glob(JF + tag + '/cases/*.json')]
    table(rows, tag)
    for title, group in split_groups(rows).items():
        print(f'\n-- {tag}: {title}: {len(group)} cases')
        keep = dict(WINDOWS)
        for a, b in PAIRS[:8]:
            pairs = [(value(r, a, 1, 10 ** 9), value(r, b, 1, 10 ** 9)) for r in group]
            diff = np.array([x - y for x, y in pairs if x is not None and y is not None])
            if len(diff) < 5:
                continue
            ci = boot(diff)
            print(f'  {a:16s} - {b:16s} {diff.mean() * 100:+6.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]  n={len(diff)}  win {int((diff > .10).sum())} lose {int((diff < -.10).sum())}')
