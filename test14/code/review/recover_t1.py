#!/usr/bin/env python3
"""Why does ridge_nc beat OLV early? Checks whether it is a reset toward the prompt, and bounds a selector."""
import glob
import json
import sys

import numpy as np

JF = '/workspace/cmmt-design-review-20261104/runs/jf/'


def boot(diff, seed=7):
    rng = np.random.default_rng(seed)
    s = diff[rng.integers(0, len(diff), (20000, len(diff)))].mean(1)
    return tuple(np.percentile(s, [2.5, 97.5]) * 100)


def value(row, method, low=1, high=10 ** 9):
    s = row['scores'][method]
    picked = [(j + f) / 2 for o, j, f, v in zip(row['offsets'], s['J'], s['F'], s['visible'])
              if low <= o <= high and j is not None and v]
    return float(np.mean(picked)) if picked else None


def area(row, method, low=1, high=10):
    s = row['scores'][method]
    picked = [a for o, a in zip(row['offsets'], s['area']) if low <= o <= high]
    return float(np.mean(picked)) * 100 if picked else float('nan')


for tag in sys.argv[1:]:
    rows = [json.load(open(f)) for f in glob.glob(JF + tag + '/cases/*.json')]
    rows = [r for r in rows if value(r, 'target_ref') is not None]
    print(f'\n===== {tag}: {len(rows)} scored cases')

    # Group cases by how Small stands at the switch, judged by its first 10 frames against GT.
    # This is a diagnostic only: a real method cannot see these scores.
    lost = [r for r in rows if (value(r, 'small', 1, 10) or 0) < .30]
    kept = [r for r in rows if (value(r, 'small', 1, 10) or 0) >= .30]
    print(f'Small already failing in +1..10 (J&F under 30): {len(lost)} cases; otherwise {len(kept)}')
    for name, group in (('Small failing', lost), ('Small fine', kept)):
        print(f' [{name}] all-frame J&F, video mean')
        for m in ('small', 'target_ref', 'orig', 'lastvis', 'olv', 'ridge_nc', 'legacy_nc', 'ridge_nc_anchor', 'direct_nc'):
            first = [value(r, m, 1, 10) for r in group]
            whole = [value(r, m) for r in group]
            first = [x for x in first if x is not None]
            whole = [x for x in whole if x is not None]
            print(f'  {m:16s} +1..10 {np.mean(first) * 100:6.2f}   all {np.mean(whole) * 100:6.2f}')

    print('\nStrongest early wins of ridge_nc over olv: J&F in +1..10 and mean predicted area %')
    print('  case          gap   small target   orig   olv  ridge | area small target orig olv ridge')
    picked = [r for r in rows if value(r, 'olv', 1, 10) is not None and value(r, 'ridge_nc', 1, 10) is not None
              and value(r, 'olv', 1, 10) - value(r, 'ridge_nc', 1, 10) < -.10]
    picked.sort(key=lambda r: value(r, 'olv', 1, 10) - value(r, 'ridge_nc', 1, 10))
    for r in picked:
        cells = [value(r, m, 1, 10) * 100 for m in ('small', 'target_ref', 'orig', 'olv', 'ridge_nc')]
        areas = [area(r, m) for m in ('small', 'target_ref', 'orig', 'olv', 'ridge_nc')]
        print(f"  {r['case']['video_id']:12s} {r['switch'] - r['last_visible']:4d}  " + ' '.join(f'{x:6.1f}' for x in cells)
              + '  | ' + ' '.join(f'{x:5.2f}' for x in areas))

    print('\nUpper bound of a per-case selector (uses GT, so not achievable)')
    pools = {'olv or ridge_nc': ('olv', 'ridge_nc'), 'olv or orig': ('olv', 'orig'), 'olv or small': ('olv', 'small'),
             'olv, ridge_nc or small': ('olv', 'ridge_nc', 'small')}
    base = np.array([value(r, 'olv') for r in rows])
    ref = np.array([value(r, 'target_ref') for r in rows])
    print(f"  olv alone {base.mean() * 100:.2f}   target_ref {ref.mean() * 100:.2f}")
    for name, members in pools.items():
        best = np.array([max(value(r, m) or 0 for m in members) for r in rows])
        ci = boot(best - ref)
        print(f'  best of {name:26s} {best.mean() * 100:6.2f}  vs target_ref {(best - ref).mean() * 100:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]')

    print('\nCases with Small mask lost at switch: does last-visible age matter?')
    gaps = np.array([r['switch'] - r['last_visible'] for r in rows])
    for low, high in ((0, 0), (1, 10), (11, 100), (101, 10 ** 9)):
        group = [r for r, g in zip(rows, gaps) if low <= g <= high]
        if len(group) < 5:
            continue
        diff = np.array([value(r, 'olv') - value(r, 'ridge_nc') for r in group])
        ci = boot(diff)
        print(f'  gap {low}..{high}: {len(group)} cases, olv - ridge_nc {diff.mean() * 100:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]')
