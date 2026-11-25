#!/usr/bin/env python3
"""J&F on trained videos against untouched ones, in the difference from Base+ on the same case. No model runs.

fit and dev are random video subsets of one pool, so the shift between them in (method - Base+) is the effect of exposure in training.
Large losses are cases that end at least 20 points under Base+.
"""
import glob
import json

import numpy as np

JF = '/workspace/cmmt-design-review-20261104/runs/jf/'
DEV_TAGS = {'mose': ('t2', 't5', 't6', 't12'), 'lvd': ('t14', 't16')}
FIT_TAGS = {'mose': 't20_mose', 'lvd': 't20_lvos'}
WINDOWS = {'all': (1, 10 ** 9), '+1..30': (1, 30), '+31..100': (31, 100)}


def rows_of(tags):
    table = {}
    for tag in tags:
        for path in glob.glob(f'{JF}{tag}/cases/*.json'):
            row = json.load(open(path))
            entry = table.setdefault(path.rsplit('/', 1)[1], dict(offsets=row['offsets'], scores={}))
            for method, score in row['scores'].items():
                entry['scores'].setdefault(method, score)
    return list(table.values())


def value(row, method, low, high):
    s = row['scores'][method]
    picked = [(j + f) / 2 for o, j, f, v in zip(row['offsets'], s['J'], s['F'], s['visible']) if low <= o <= high and j is not None and v]
    return float(np.mean(picked)) if picked else None


def diffs(rows, a, b, low, high):
    out = []
    for r in rows:
        if a in r['scores'] and b in r['scores']:
            x, y = value(r, a, low, high), value(r, b, low, high)
            if x is not None and y is not None:
                out.append(x - y)
    return np.array(out) * 100


def boot(a, b=None, seed=7):
    rng = np.random.default_rng(seed)
    s = a[rng.integers(0, len(a), (20000, len(a)))].mean(1)
    mean = a.mean()
    if b is not None:
        s = s - b[rng.integers(0, len(b), (20000, len(b)))].mean(1)
        mean = a.mean() - b.mean()
    low, high = np.percentile(s, [2.5, 97.5])
    return f'{mean:+6.2f} [{low:+6.2f},{high:+6.2f}]'


def level(rows, method):
    got = [value(r, method, 1, 10 ** 9) for r in rows if method in r['scores']]
    got = [x for x in got if x is not None]
    return f'{np.mean(got) * 100:6.2f}' if got else '     -'


def main():
    for key, tags in (('mose', [f'{t}_mose' for t in DEV_TAGS['mose']]), ('lvd', [f'{t}_lvd' for t in DEV_TAGS['lvd']])):
        fit, dev = rows_of([FIT_TAGS[key]]), rows_of(tags)
        print(f'\n===== {key}: trained cases {len(fit)}, untouched cases {len(dev)}')
        for method in ('target_ref', 'mlp_context', 'e2e3_ctx_f'):
            print(f'  {method:12s} trained {level(fit, method)}  untouched {level(dev, method)}')
        for a, b in (('e2e3_ctx_f', 'target_ref'), ('mlp_context', 'target_ref'), ('e2e3_ctx_f', 'mlp_context')):
            for name, (low, high) in WINDOWS.items():
                x, y = diffs(fit, a, b, low, high), diffs(dev, a, b, low, high)
                if len(x) < 3 or len(y) < 3:
                    print(f'  {a} - {b} {name}: trained n={len(x)} untouched n={len(y)}')
                    continue
                print(f'  {a:11s} - {b:11s} {name:9s} trained {boot(x)} n={len(x):3d}  untouched {boot(y)} n={len(y):3d}  shift {boot(x, y)}')
        for a, b in (('e2e3_ctx_f', 'target_ref'), ('mlp_context', 'target_ref')):
            x, y = diffs(fit, a, b, 1, 10 ** 9), diffs(dev, a, b, 1, 10 ** 9)
            if len(x) >= 3 and len(y) >= 3:
                print(f'  large losses {a:11s}: trained {int((x <= -20).sum())}/{len(x)} ({np.mean(x <= -20) * 100:.1f}%), untouched {int((y <= -20).sum())}/{len(y)} ({np.mean(y <= -20) * 100:.1f}%)'
                      f'; large wins trained {int((x >= 20).sum())}, untouched {int((y >= 20).sum())}')


if __name__ == '__main__':
    main()
