#!/usr/bin/env python3
"""Paired comparisons for full translation. Methods may come from different runs of the same cases.

Usage: paired_acc.py <dataset key> ...   where a key is val or mose.
"""
import glob
import json
import os
import sys

import numpy as np

JF = '/workspace/cmmt-design-review-20261104/runs/jf/'
WINDOWS = {'all': (1, 10 ** 9), '+1..30': (1, 30), '+31..100': (31, 100), '+101..': (101, 10 ** 9)}
PAIRS = [('mlp', 'legacy'), ('mlp_context', 'legacy'), ('transformer', 'legacy'), ('ds3', 'legacy'), ('ds_context', 'legacy'), ('ridge', 'legacy'),
         ('ridge_gain', 'ridge'), ('g110', 'legacy'), ('g120', 'legacy'), ('g130', 'legacy'), ('g150', 'legacy'),
         ('swap1', 'legacy'), ('swap7', 'legacy'), ('swap30', 'legacy'), ('pin7', 'legacy'), ('swap7g', 'legacy'),
         ('swap7_ridge', 'ridge'), ('swap7_mlp', 'mlp'), ('swap7_ctx', 'mlp_context'), ('swap7', 'swap7_ridge'), ('swap7_mlp', 'swap7'), ('swap7_ctx', 'swap7'),
         ('legacy', 'target_ref'), ('swap7', 'target_ref'), ('swap30', 'target_ref'), ('g120', 'target_ref'), ('swap7_ctx', 'target_ref'),
         ('cond_err', 'target_ref'), ('err_near', 'target_ref'), ('err_far', 'target_ref'), ('err_low', 'target_ref'), ('err_high', 'target_ref'),
         ('err_shrink', 'target_ref'), ('err_rest', 'target_ref'),
         ('m110_mlp', 'mlp'), ('m110_ctx', 'mlp_context'), ('m110_ds3', 'ds3'), ('m110_dsc', 'ds_context'),
         ('m110_mlp', 'legacy'), ('m110_ctx', 'legacy'), ('m110_ds3', 'legacy'), ('m110_dsc', 'legacy'),
         ('gc110', 'legacy'), ('gr110', 'legacy'), ('band_em', 'legacy'), ('band_h', 'legacy'), ('cal_leg', 'legacy'),
         ('cal_ctx', 'mlp_context'), ('band_em', 'g110'), ('cal_leg', 'g110'),
         ('m110_ctx', 'target_ref'), ('m110_mlp', 'target_ref'),
         ('e2e_mlp', 'mlp'), ('e2e_ctx', 'mlp_context'), ('e2e_mlp_c', 'mlp'), ('e2e_ctx_c', 'mlp_context'),
         ('e2e_mlp', 'legacy'), ('e2e_ctx', 'legacy'), ('e2e_mlp', 'target_ref'), ('e2e_ctx', 'target_ref'),
         ('e2e_ctx3k', 'mlp_context'), ('e2e_ctx3k_c', 'mlp_context'), ('e2e_ctx3k', 'legacy'), ('e2e_ctx3k', 'g110'), ('e2e_ctx3k', 'target_ref'),
         ('e2e_ctx_f', 'e2e_ctx3k'), ('e2e_ctx_f', 'mlp_context'), ('e2e_ctx_fc', 'mlp_context'), ('e2e_ctx_f', 'legacy'), ('e2e_ctx_f', 'target_ref'),
         ('e2e_ctx_pin7', 'e2e_ctx_f'), ('e2e_ctx_pin7', 'target_ref'), ('m110_ctx', 'e2e_ctx_f'),
         ('e2e_mlp_f', 'mlp'), ('e2e_mlp_f', 'e2e_ctx_f'), ('e2e_mlp_f', 'target_ref'),
         ('e2e2_ctx_f', 'e2e_ctx_f'), ('e2e2_ctx_f', 'target_ref'),
         ('e2e_ctx_pin1', 'e2e_ctx_f'), ('e2e_ctx_pin2', 'e2e_ctx_f'), ('e2e_ctx_pin3', 'e2e_ctx_f'), ('e2e_ctx_pin5', 'e2e_ctx_f'),
         ('e2e_ctx_pin15', 'e2e_ctx_f'), ('e2e_ctx_pin30', 'e2e_ctx_f'), ('e2e_ctx_pin3', 'e2e_ctx_pin7'), ('e2e_ctx_pin1', 'target_ref'), ('e2e_ctx_pin3', 'target_ref'),
         ('e2e_ctx_pin5', 'target_ref'), ('e2e_ctx_pin7', 'target_ref'), ('e2e3_dsc_f', 'e2e_ctx_f'), ('e2e3_dsc_f', 'e2e3_ctx_f'), ('e2e3_dsc_f', 'target_ref'),
         ('e2e_tf_f', 'transformer'), ('e2e_tf_f', 'e2e_ctx_f'), ('e2e_tf_f', 'e2e3_ctx_f'), ('e2e_tf_f', 'target_ref'), ('e2e_tf_f', 'legacy'),
         ('e2e_mlp_f', 'e2e3_ctx_f'), ('e2e3_ctx_f', 'legacy'), ('e2e3_ctx_f', 'mlp_context'),
         ('e2e3_ctx_pin3', 'e2e3_ctx_f'), ('e2e3_ctx_pin5', 'e2e3_ctx_f'), ('e2e3_ctx_pin7', 'e2e3_ctx_f'),
         ('e2e3_ctx_pin3', 'target_ref'), ('e2e3_ctx_pin5', 'target_ref'), ('e2e3_ctx_pin7', 'target_ref'),
         ('e2e3_ctx_f', 'e2e_ctx_f'), ('e2e3_ctx_f', 'e2e2_ctx_f'), ('e2e3_ctx_f', 'target_ref')]


def boot(diff, seed=7):
    rng = np.random.default_rng(seed)
    s = diff[rng.integers(0, len(diff), (20000, len(diff)))].mean(1)
    return tuple(np.percentile(s, [2.5, 97.5]) * 100)


def value(row, method, low, high):
    s = row['scores'][method]
    picked = [(j + f) / 2 for o, j, f, v in zip(row['offsets'], s['J'], s['F'], s['visible'])
              if low <= o <= high and j is not None and v]
    return float(np.mean(picked)) if picked else None


def load(key):
    table = {}
    for tag in ('t2', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't4', 't1'):
        for path in glob.glob(f'{JF}{tag}_{key}/cases/*.json'):
            row = json.load(open(path))
            entry = table.setdefault(os.path.basename(path), dict(offsets=row['offsets'], scores={}))
            for method, score in row['scores'].items():
                entry['scores'].setdefault(method, score)
    return table


for key in sys.argv[1:]:
    table = load(key)
    print(f'\n===== {key}: {len(table)} cases seen')
    names = sorted({m for r in table.values() for m in r['scores']})
    for method in names:
        got = [value(r, method, 1, 10 ** 9) for r in table.values() if method in r['scores']]
        got = [x for x in got if x is not None]
        print(f'  {method:18s} {np.mean(got) * 100:6.2f}  n={len(got)}')
    for name, (low, high) in WINDOWS.items():
        print(f' [{name}]')
        for a, b in PAIRS:
            pairs = [(value(r, a, low, high), value(r, b, low, high)) for r in table.values() if a in r['scores'] and b in r['scores']]
            diff = np.array([x - y for x, y in pairs if x is not None and y is not None])
            if len(diff) < 8:
                continue
            ci = boot(diff)
            print(f'  {a:14s} - {b:12s} {diff.mean() * 100:+6.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]  n={len(diff)}  win {int((diff > .10).sum())} lose {int((diff < -.10).sum())}')
