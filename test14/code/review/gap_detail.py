#!/usr/bin/env python3
"""Where the gap between trained and untouched videos comes from, using the items gap_proxy.py stored. No model runs.

Rounds are read on the same items, so each step of fine-tuning is a paired difference. Failures are items with IoU below .5.
MOSE dev is split by whether the video shares a scene with training (dup_scan.json, correlation .97, share .25).
"""
import json

import numpy as np

ROOT = '/workspace/cmmt-design-review-20261104/runs/analysis/'
GROUPS = ('mose_fit', 'mose_dev', 'lvos_fit', 'lvos_dev', 'lvos_val')
CHAINS = (('mlp_context', 'e2e_ctx_last', 'e2e3_ctx'), ('mlp', 'e2e_mlp_last'), ('transformer', 'e2e_tf'))


def boot(d, seed=7):
    rng = np.random.default_rng(seed)
    s = d[rng.integers(0, len(d), (20000, len(d)))].mean(1)
    return (d.mean(), *np.percentile(s, [2.5, 97.5]))


def text(d):
    mean, low, high = boot(np.asarray(d, dtype=float))
    return f'{mean:+.4f} [{low:+.4f},{high:+.4f}]'


def main():
    data = json.load(open(ROOT + 'gap_proxy.json'))
    table, videos = data['table'], data['videos']
    scan = json.load(open(ROOT + 'dup_scan.json'))
    flagged = {n for rows in scan.values() for n, v in rows.items() if v['shares']['0.97'] >= .25}
    for scenario in ('alone', 'cond'):
        print(f'\n===== {scenario}')
        print(f"  {'failures, IoU<.5':16s} " + ' '.join(f'{g:>10s}' for g in GROUPS) + '     median IoU ' + ' '.join(f'{g:>9s}' for g in GROUPS))
        for chain in CHAINS:
            for model in chain:
                rates = [np.mean(np.array(table[g][model][scenario]['iou']) < .5) for g in GROUPS]
                medians = [np.median(table[g][model][scenario]['iou']) for g in GROUPS]
                print(f'  {model:16s} ' + ' '.join(f'{r:10.3f}' for r in rates) + '                ' + ' '.join(f'{m:9.4f}' for m in medians))
        print('  each round on the same items, IoU then feature error')
        for chain in CHAINS:
            for before, after in zip(chain, chain[1:]):
                for group in GROUPS:
                    a = np.array(table[group][after][scenario]['iou']) - np.array(table[group][before][scenario]['iou'])
                    b = np.array(table[group][after][scenario]['feature']) - np.array(table[group][before][scenario]['feature'])
                    print(f'    {after:13s} - {before:13s} {group:9s} iou {text(a)}  feature {text(b)}')
        print('  MOSE dev split by shared scene, gain of fine-tuning over the same design before it')
        shared = np.array([v in flagged for v in videos['mose_dev']])
        print(f'    shared {int(shared.sum())}, clean {int((~shared).sum())}; mose_fit for reference')
        for tuned, plain in (('e2e3_ctx', 'mlp_context'), ('e2e_ctx_last', 'mlp_context'), ('e2e_mlp_last', 'mlp'), ('e2e_tf', 'transformer')):
            gain = np.array(table['mose_dev'][tuned][scenario]['iou']) - np.array(table['mose_dev'][plain][scenario]['iou'])
            fit = np.array(table['mose_fit'][tuned][scenario]['iou']) - np.array(table['mose_fit'][plain][scenario]['iou'])
            print(f'    {tuned:13s} fit {text(fit)}  shared {text(gain[shared])}  clean {text(gain[~shared])}')
        print('  level by group for the shared split, IoU')
        for model in ('mlp_context', 'e2e_ctx_last', 'e2e3_ctx'):
            iou = np.array(table['mose_dev'][model][scenario]['iou'])
            print(f"    {model:13s} fit {np.mean(table['mose_fit'][model][scenario]['iou']):.4f}  shared {iou[shared].mean():.4f}  clean {iou[~shared].mean():.4f}")


if __name__ == '__main__':
    main()
