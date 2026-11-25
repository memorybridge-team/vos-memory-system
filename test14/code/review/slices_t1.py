#!/usr/bin/env python3
"""Event slices from the GT timeline (doc 03 section 4), false positives, recovery, and the early deficit of OLV.

Slices use only the GT timeline after the switch, never a method's score, so they do not
select on the outcome being compared.
"""
import glob
import json
import sys

import numpy as np

JF = '/workspace/cmmt-design-review-20261104/runs/jf/'
ALL = (1, 10 ** 9)
PAIRS = [('target_ref', 'small'), ('olv', 'small'), ('ridge_nc', 'small'), ('legacy_nc', 'small'),
         ('olv', 'ridge_nc'), ('olv', 'legacy_nc'), ('ridge_nc_anchor', 'olv')]


def boot(diff, seed=7):
    rng = np.random.default_rng(seed)
    s = diff[rng.integers(0, len(diff), (20000, len(diff)))].mean(1)
    return tuple(np.percentile(s, [2.5, 97.5]) * 100)


def value(row, method, low, high):
    s = row['scores'][method]
    picked = [(j + f) / 2 for o, j, f, v in zip(row['offsets'], s['J'], s['F'], s['visible'])
              if low <= o <= high and j is not None and v]
    return float(np.mean(picked)) if picked else None


def timeline(row):
    return [(o, v) for o, v in zip(row['offsets'], row['scores']['target_ref']['visible']) if v is not None]


def tags(row):
    line = timeline(row)
    if not line:
        return {'no-annotation'}
    seen = [v for _, v in line]
    result = set()
    gone = any(a and not b for a, b in zip(seen, seen[1:]))
    back = any(b and not a for a, b in zip(seen, seen[1:]))
    if not any(seen):
        result.add('never-visible')
    elif all(seen):
        result.add('continuous-visible')
    if not seen[0]:
        result.add('absent-at-switch')
    if gone:
        result.add('disappearance')
    if back:
        result.add('reappearance')
    if gone and not seen[-1]:
        result.add('no-return')
    return result


def report(rows, title, pairs=PAIRS):
    print(f'\n-- {title}: {len(rows)} cases')
    for a, b in pairs:
        got = [(value(r, a, *ALL), value(r, b, *ALL)) for r in rows]
        diff = np.array([x - y for x, y in got if x is not None and y is not None])
        if len(diff) < 8:
            continue
        ci = boot(diff)
        print(f'  {a:16s} - {b:12s} {diff.mean() * 100:+6.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]  n={len(diff)}  win {int((diff > .10).sum())} lose {int((diff < -.10).sum())}')


def false_positive(rows, methods):
    print('\nGT-absent frames: share with any predicted foreground (case mean)')
    for method in methods:
        rates, fractions = [], []
        for r in rows:
            s = r['scores'][method]
            area = [a for a, v in zip(s['area'], s['visible']) if v is False]
            if area:
                rates.append(np.mean([x > 0 for x in area]))
                fractions.append(np.mean(area))
        if rates:
            print(f'  {method:16s} rate {np.mean(rates) * 100:6.2f}%  mean area {np.mean(fractions) * 100:.3f}%  n={len(rates)}')


def recovery(rows, methods):
    print('\nFirst reappearance after a GT-absent stretch: J&F of the first 1 / 5 / 20 visible annotated frames, and never-recovered share')
    for method in methods:
        first, five, twenty, lost = [], [], [], []
        for r in rows:
            s = r['scores'][method]
            line = [(i, v) for i, v in enumerate(s['visible']) if v is not None]
            start = next((b for (a, x), (b, y) in zip(line, line[1:]) if not x and y), None)
            if start is None:
                continue
            after = [(s['J'][i] + s['F'][i]) / 2 for i, v in line if i >= start and v]
            first.append(after[0])
            five.append(np.mean(after[:5]))
            twenty.append(np.mean(after[:20]))
            lost.append(np.max([s['J'][i] for i, v in line if i >= start and v][:20]) < .1)
        if first:
            print(f'  {method:16s} +1 {np.mean(first) * 100:6.2f}  first5 {np.mean(five) * 100:6.2f}  first20 {np.mean(twenty) * 100:6.2f}  no-recovery {np.mean(lost) * 100:5.1f}%  n={len(first)}')


def deficit(rows, name):
    print(f'\nEarly deficit, {name}: cases where olv falls more than 10 points below ridge_nc in +1..10')
    print('  case          last-gap  small  target  ridge_nc  olv  lastvis  | small@+1  olv@+1  ridge@+1')
    picked = []
    for r in rows:
        a, b = value(r, 'olv', 1, 10), value(r, 'ridge_nc', 1, 10)
        if a is not None and b is not None and a - b < -.10:
            picked.append(r)
    picked.sort(key=lambda r: value(r, 'olv', 1, 10) - value(r, 'ridge_nc', 1, 10))

    def at(row, method):
        s = row['scores'][method]
        for j, v in zip(s['J'], s['visible']):
            if j is not None and v:
                return j
        return float('nan')

    for r in picked:
        cells = [value(r, m, 1, 10) for m in ('small', 'target_ref', 'ridge_nc', 'olv', 'lastvis')]
        print(f"  {r['case']['video_id']:12s} {r['switch'] - r['last_visible']:8d}  " + '  '.join(f'{x * 100:5.1f}' for x in cells)
              + f"  | {at(r, 'small'):.2f} {at(r, 'olv'):.2f} {at(r, 'ridge_nc'):.2f}")
    gaps = np.array([r['switch'] - r['last_visible'] for r in rows])
    print(f'  last visible equals switch in {int((gaps == 0).sum())}/{len(gaps)} cases; gap over 10 frames in {int((gaps > 10).sum())}')
    same = np.mean([np.array_equal(np.array(r['scores']['recent_ridge']['J'], dtype=float), np.array(r['scores']['lastvis']['J'], dtype=float)) for r in rows])
    frames = [(x != y) for r in rows for x, y in zip(r['scores']['recent_ridge']['area'], r['scores']['lastvis']['area'])]
    print(f'  recent_ridge vs lastvis: identical J timeline in {same * 100:.1f}% of cases; area differs on {np.mean(frames) * 100:.1f}% of frames')
    same = np.mean([np.array_equal(np.array(r['scores']['ridge_nc_anchor']['J'], dtype=float), np.array(r['scores']['olv']['J'], dtype=float)) for r in rows])
    frames = [(x != y) for r in rows for x, y in zip(r['scores']['ridge_nc_anchor']['area'], r['scores']['olv']['area'])]
    print(f'  ridge_nc_anchor vs olv: identical J timeline in {same * 100:.1f}% of cases; area differs on {np.mean(frames) * 100:.1f}% of frames')


METHODS = ('target_ref', 'small', 'legacy_nc', 'ridge_nc', 'orig', 'lastvis', 'olv', 'ridge_nc_anchor')
for tag in sys.argv[1:]:
    rows = [json.load(open(f)) for f in glob.glob(JF + tag + '/cases/*.json')]
    print(f'\n===== {tag}: {len(rows)} cases')
    tagged = [(r, tags(r)) for r in rows]
    names = sorted({t for _, ts in tagged for t in ts})
    print({n: sum(n in ts for _, ts in tagged) for n in names})
    for name in ('continuous-visible', 'absent-at-switch', 'disappearance', 'reappearance', 'no-return'):
        group = [r for r, ts in tagged if name in ts]
        if group:
            report(group, name)
    hard = [r for r, ts in tagged if ts & {'absent-at-switch', 'disappearance', 'reappearance'}]
    report(hard, 'any absence event after the switch')
    false_positive(rows, METHODS)
    recovery(rows, METHODS)
    deficit(rows, tag)
