#!/usr/bin/env python3
"""Paired comparisons on the external datasets, across runs of the same switch cases. Video-clustered. No model runs.

Usage: unseen_pair.py <dataset key> ...   where a key is m3vos or pumavos; tags 3, 4 and 5 are merged by video, object and switch.
"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

ROOT = '/workspace/cmmt-design-review-20261104/runs/unseen/'
PAIRS = (('e2e_mlp_f', 'e2e_ctx_f'), ('e2e_tf_f', 'e2e_ctx_f'), ('e2e_mlp_f', 'base'), ('e2e_tf_f', 'base'), ('e2e_mlp_f', 'legacy'), ('e2e_tf_f', 'legacy'),
         ('e2e3_ctx_f', 'e2e_ctx_f'), ('e2e3_ctx_f', 'mlp_context'), ('e2e3_ctx_f', 'legacy'), ('e2e_ctx_f', 'mlp_context'), ('e2e_ctx_f', 'legacy'),
         ('e2e3_ctx_pin5', 'e2e3_ctx_f'), ('e2e_ctx_pin7', 'e2e_ctx_f'), ('e2e3_ctx_pin5', 'e2e_ctx_pin7'),
         ('e2e3_ctx_f', 'base'), ('e2e_ctx_f', 'base'), ('e2e3_ctx_pin5', 'base'), ('e2e_ctx_pin7', 'base'), ('mlp_context', 'base'), ('legacy', 'base'), ('small', 'base'))


def load(key):
    table = {}
    for tag in (f'{key}3', f'{key}4', f'{key}5'):
        for path in glob.glob(f'{ROOT}{tag}/cases/*.json'):
            case = json.load(open(path))
            for name, row in case['switches'].items():
                entry = table.setdefault((case['video'], case['object'], name), {})
                for method, score in row.items():
                    if isinstance(score, dict) and method != 'trace':
                        entry.setdefault(method, score)
    return table


def per_video(table, method, keys):
    grouped = defaultdict(list)
    for key in keys:
        value = table[key].get(method, {}).get('JF')
        if value is not None:
            grouped[key[0]].append(value)
    return {k: float(np.mean(v)) for k, v in grouped.items()}


def main():
    for key in sys.argv[1:]:
        table = load(key)
        print(f'\n===== {key}: {len(table)} switch cases, {len({k[0] for k in table})} videos')
        common = [k for k, row in table.items() if 'e2e3_ctx_f' in row and 'e2e_ctx_f' in row]
        print(f'  cases with both rounds {len(common)}, videos {len({k[0] for k in common})}')
        for method in ('base', 'small', 'legacy', 'mlp_context', 'e2e_mlp_f', 'e2e_tf_f', 'e2e_ctx_f', 'e2e3_ctx_f', 'e2e_ctx_pin7', 'e2e3_ctx_pin5'):
            got = per_video(table, method, common)
            if got:
                print(f'  {method:14s} {np.mean(list(got.values())) * 100:6.2f}')
        rng = np.random.default_rng(7)
        for a, b in PAIRS:
            x, y = per_video(table, a, common), per_video(table, b, common)
            shared = sorted(set(x) & set(y))
            if len(shared) < 3:
                continue
            d = np.array([x[k] - y[k] for k in shared]) * 100
            s = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
            low, high = np.percentile(s, [2.5, 97.5])
            print(f'  {a:14s} - {b:13s} {d.mean():+6.2f} [{low:+6.2f},{high:+6.2f}]  n={len(d)}  win {int((d > 1).sum())} lose {int((d < -1).sum())}')


if __name__ == '__main__':
    main()
