#!/usr/bin/env python3
"""J&F with the videos that share a scene with training removed, and the same differences on those videos alone.

Usage: clean_report.py <level> <share>   a video is flagged when at least <share> of its usable frames reach correlation <level>.
No model runs; this only reads finished cases.
"""
import glob
import json
import sys

import numpy as np

JF = '/workspace/cmmt-design-review-20261104/runs/jf/'
SCAN = '/workspace/cmmt-design-review-20261104/runs/analysis/dup_scan.json'
LEVEL, SHARE = sys.argv[1] if len(sys.argv) > 1 else '.97', float(sys.argv[2]) if len(sys.argv) > 2 else .25
LEVEL = str(float(LEVEL))
TAGS = ('t2', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't4', 't1')
METHODS = ('legacy', 'mlp', 'transformer', 'mlp_context', 'e2e_mlp_f', 'e2e_tf_f', 'e2e_ctx_f', 'e2e3_mlp_f', 'e2e3_tf_f', 'e2e3_ctx_f', 'e2e_ctx_pin7', 'e2e3_ctx_pin5')
BASES = ('target_ref', 'legacy', 'e2e_ctx_f')


def boot(diff, seed=7):
    rng = np.random.default_rng(seed)
    s = diff[rng.integers(0, len(diff), (20000, len(diff)))].mean(1)
    return tuple(np.percentile(s, [2.5, 97.5]) * 100)


def value(row, method):
    s = row['scores'][method]
    picked = [(j + f) / 2 for o, j, f, v in zip(row['offsets'], s['J'], s['F'], s['visible']) if o >= 1 and j is not None and v]
    return float(np.mean(picked)) if picked else None


def load(key):
    table = {}
    for tag in TAGS:
        for path in glob.glob(f'{JF}{tag}_{key}/cases/*.json'):
            row = json.load(open(path))
            entry = table.setdefault(path.rsplit('/', 1)[1], dict(video=row['case']['video_id'], offsets=row['offsets'], scores={}))
            for method, score in row['scores'].items():
                entry['scores'].setdefault(method, score)
    return list(table.values())


def line(rows, method, base):
    pairs = [(value(r, method), value(r, base)) for r in rows if method in r['scores'] and base in r['scores']]
    pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
    if len(pairs) < 3:
        return f'n={len(pairs)}'
    d = np.array([a - b for a, b in pairs])
    low, high = boot(d)
    return f'{d.mean() * 100:+6.2f} [{low:+6.2f},{high:+6.2f}] n={len(d)}'


def level(rows, method):
    got = [value(r, method) for r in rows if method in r['scores']]
    got = [x for x in got if x is not None]
    return f'{np.mean(got) * 100:6.2f}' if got else '     -'


def contrast(flagged, clean, method, base):
    """Difference between the two groups in one paired difference, by resampling each group."""
    def diffs(rows):
        pairs = [(value(r, method), value(r, base)) for r in rows if method in r['scores'] and base in r['scores']]
        return np.array([a - b for a, b in pairs if a is not None and b is not None])
    a, b = diffs(flagged), diffs(clean)
    if len(a) < 3 or len(b) < 3:
        return 'too few'
    rng = np.random.default_rng(7)
    s = a[rng.integers(0, len(a), (20000, len(a)))].mean(1) - b[rng.integers(0, len(b), (20000, len(b)))].mean(1)
    low, high = np.percentile(s, [2.5, 97.5]) * 100
    return f'{(a.mean() - b.mean()) * 100:+6.2f} [{low:+6.2f},{high:+6.2f}] (flagged n={len(a)}, clean n={len(b)})'


def main():
    scan = json.load(open(SCAN))
    flagged = {n for rows in scan.values() for n, v in rows.items() if v['shares'][LEVEL] >= SHARE}
    print(f'flagged at correlation {LEVEL}, share {SHARE}: ' + ', '.join(f"{g} {sum(v['shares'][LEVEL] >= SHARE for v in rows.values())}/{len(rows)}" for g, rows in scan.items()))
    for key in sys.argv[3:] or ('mose', 'val', 'lvd'):
        rows = load(key)
        a, b = [r for r in rows if r['video'] in flagged], [r for r in rows if r['video'] not in flagged]
        print(f'\n===== {key}: {len(rows)} cases, scene shared with training {len(a)}, clean {len(b)}')
        print(f"  {'method':14s} {'all':>6s} {'clean':>6s} {'shared':>6s}")
        for method in ('target_ref',) + METHODS:
            print(f'  {method:14s} {level(rows, method)} {level(b, method)} {level(a, method)}')
        for base in BASES:
            print(f'  differences against {base}, clean videos, then all')
            for method in METHODS:
                if method != base:
                    print(f'    {method:14s} {line(b, method, base)}   all {line(rows, method, base)}')
        print('  shared minus clean, in the difference from Base+ and in the gain over the same design before fine-tuning')
        for method, base in (('e2e3_ctx_f', 'target_ref'), ('e2e_ctx_f', 'target_ref'), ('mlp_context', 'target_ref'), ('legacy', 'target_ref'), ('e2e3_ctx_f', 'mlp_context'), ('e2e_ctx_f', 'mlp_context')):
            print(f'    {method:12s} - {base:12s} {contrast(a, b, method, base)}')


if __name__ == '__main__':
    main()
