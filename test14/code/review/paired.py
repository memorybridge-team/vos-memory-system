#!/usr/bin/env python3
"""Paired method-vs-method comparisons and the reproducibility floor of the J&F rollouts."""
import glob
import json
from collections import defaultdict

import numpy as np

JF = '/workspace/cmmt-design-review-20261104/runs/jf/'
UN = '/workspace/cmmt-design-review-20261104/runs/unseen/'


def boot(diff, seed=7):
    rng = np.random.default_rng(seed)
    s = diff[rng.integers(0, len(diff), (20000, len(diff)))].mean(1)
    return tuple(np.percentile(s, [2.5, 97.5]) * 100)


def case_value(row, method):
    s = row['scores'][method]
    picked = [(j + f) / 2 for j, f, v in zip(s['J'], s['F'], s['visible']) if j is not None and v]
    return float(np.mean(picked)) if picked else None


def load_jf(tag):
    return [json.load(open(f)) for f in glob.glob(JF + tag + '/cases/*.json')]


def compare_jf(tag, pairs):
    rows = load_jf(tag)
    print(f'\n== {tag} ({len(rows)} cases)')
    for a, b in pairs:
        d = {}
        for r in rows:
            x, y = case_value(r, a), case_value(r, b)
            if x is not None and y is not None:
                d[r['case']['video_id']] = x - y
        diff = np.array(list(d.values()))
        low, high = boot(diff)
        print(f'  {a:20s} - {b:20s} {diff.mean() * 100:+6.2f} [{low:+.2f},{high:+.2f}]  n={len(diff)}  win {int((diff > .10).sum())} lose {int((diff < -.10).sum())}')


def compare_unseen(tag, pairs, key='JF'):
    cases = [json.load(open(f)) for f in glob.glob(UN + tag + '/cases/*.json')]
    print(f'\n== {tag} ({len(cases)} objects) {key}')
    for a, b in pairs:
        grouped = defaultdict(list)
        for c in cases:
            for row in c['switches'].values():
                x, y = row.get(a, {}).get(key), row.get(b, {}).get(key)
                if x is not None and y is not None:
                    grouped[c['video']].append(x - y)
        diff = np.array([np.mean(v) for v in grouped.values()])
        low, high = boot(diff)
        print(f'  {a:20s} - {b:20s} {diff.mean() * 100:+6.2f} [{low:+.2f},{high:+.2f}]  n={len(diff)}')


compare_jf('sens_val', [('legacy_cond_oracle', 'legacy'), ('legacy_ptr_only', 'legacy'), ('legacy_recent_oracle', 'legacy'), ('legacy', 'ridge')])
compare_jf('sens_mose', [('legacy_cond_oracle', 'legacy'), ('legacy_ptr_only', 'legacy'), ('legacy_recent_oracle', 'legacy'), ('legacy', 'ridge')])
compare_jf('cand_val', [('legacy_nc', 'ridge_nc'), ('transformer_nc', 'ridge_nc'), ('mlp_nc', 'ridge_nc'), ('mlp_context_nc', 'ridge_nc'), ('ds3_nc', 'ridge_nc'), ('conv3mlp_nc', 'ridge_nc')])
PAIRS = [('legacy_nc', 'legacy'), ('ridge_nc', 'ridge'), ('legacy', 'ridge'), ('legacy_nc', 'ridge_nc'), ('olv', 'ridge_nc'), ('olv', 'legacy_nc'), ('olv', 'ridge_nc_anchor'), ('olv', 'ridge_anchor'), ('ridge_nc_anchor', 'ridge_anchor')]
compare_unseen('pumavos', PAIRS)
compare_unseen('m3vos', PAIRS)

# Same case, same target state, two runs: how far does the reference itself move?
print('\n== reproducibility of the target-state reference, sens_val vs cand_val')
a = {r['case']['case_id']: r for r in load_jf('sens_val')}
b = {r['case']['case_id']: r for r in load_jf('cand_val')}
diffs, env211, env28 = [], [], []
for key in sorted(set(a) & set(b)):
    x, y = case_value(a[key], 'target_ref'), case_value(b[key], 'target_ref')
    if x is None or y is None:
        continue
    diffs.append(x - y)
diffs = np.array(diffs)
print(f'  shared cases {len(diffs)}  exact {int((diffs == 0).sum())}  |d|<0.5pt {int((abs(diffs) < .005).sum())}  |d|>2pt {int((abs(diffs) > .02).sum())}  |d|>10pt {int((abs(diffs) > .10).sum())}  max {abs(diffs).max() * 100:.1f}  mean {diffs.mean() * 100:+.2f}')
for key in sorted(set(a) & set(b)):
    x, y = case_value(a[key], 'target_ref'), case_value(b[key], 'target_ref')
    if x is not None and y is not None and abs(x - y) > .10:
        print('   ', key[:60], round(x * 100, 1), round(y * 100, 1))
