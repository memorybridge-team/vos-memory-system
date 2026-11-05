#!/usr/bin/env python3
"""Do fine-tuned models read better on videos they trained on? The same measurement on trained and untouched cases.

fit and dev are random video subsets of one official training pool, so a difference between them comes from exposure during training.
LVOS val is official validation, a different pool, so its level is not comparable; the fine-tuning gain inside it is.
One case per video, offsets as in held_out, last weights of each model. Translated pointers come from the ridge, as in the third recipe.
Writes runs/analysis/gap_proxy.json.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from jf_eval import ROOT, select_cases
from train_e2e import OFFSETS, Store, Teacher, fixed_translators, one, ridge_pointer

GROUPS = (('mose_fit', 'fit', 'MOSEv2', 300), ('mose_dev', 'dev', 'MOSEv2', 300), ('lvos_fit', 'fit', 'LVOS v2', 150),
          ('lvos_dev', 'dev', 'LVOS v2', 73), ('lvos_val', 'val', 'LVOS v2', 140))
MODELS = ('mlp_context', 'e2e_ctx_last', 'e2e3_ctx', 'mlp', 'e2e_mlp_last', 'transformer', 'e2e_tf')
FIELDS = ('iou', 'feature', 'state', 'lost', 'extra')
SEED = 11


def boot(a, b, seed=7):
    rng = np.random.default_rng(seed)
    x = a[rng.integers(0, len(a), (20000, len(a)))].mean(1) - b[rng.integers(0, len(b), (20000, len(b)))].mean(1)
    return (a.mean() - b.mean(), *np.percentile(x, [2.5, 97.5]))


def report(table):
    for scenario in ('alone', 'cond'):
        print(f'\n[{scenario}] mean IoU with the Base+ read, and feature error', flush=True)
        print(f"  {'model':14s} " + ' '.join(f'{g:>16s}' for g, *_ in GROUPS))
        for model in MODELS:
            cells = [table[g][model][scenario] for g, *_ in GROUPS]
            print(f'  {model:14s} ' + ' '.join(f"{np.mean(c['iou']):8.4f} {np.mean(c['feature']):7.5f}" for c in cells))
        for field in ('iou', 'feature'):
            print(f'  trained minus untouched, {field}; better on trained means iou above 0 or feature below 0')
            for model in MODELS:
                row = []
                for a, b in (('mose_fit', 'mose_dev'), ('lvos_fit', 'lvos_dev')):
                    mean, low, high = boot(np.array(table[a][model][scenario][field]), np.array(table[b][model][scenario][field]))
                    row.append(f'{a[:4]} {mean:+.4f} [{low:+.4f},{high:+.4f}]')
                print(f'    {model:14s} ' + '  '.join(row))
        print('  gain of fine-tuning on the trained cases minus the same gain on untouched cases, iou')
        for tuned, plain in (('e2e3_ctx', 'mlp_context'), ('e2e_ctx_last', 'mlp_context'), ('e2e_mlp_last', 'mlp'), ('e2e_tf', 'transformer')):
            row = []
            for a, b in (('mose_fit', 'mose_dev'), ('lvos_fit', 'lvos_dev')):
                gain_a = np.array(table[a][tuned][scenario]['iou']) - np.array(table[a][plain][scenario]['iou'])
                gain_b = np.array(table[b][tuned][scenario]['iou']) - np.array(table[b][plain][scenario]['iou'])
                mean, low, high = boot(gain_a, gain_b)
                row.append(f'{a[:4]} fit {gain_a.mean():+.4f} dev {gain_b.mean():+.4f} diff {mean:+.4f} [{low:+.4f},{high:+.4f}]')
            print(f'    {tuned:14s} ' + '  '.join(row))


@torch.no_grad()
def main():
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(.25)
    device = torch.device('cuda')
    teacher = Teacher(device)
    encode = teacher.image
    memo = {}

    def image(tensor):
        if memo.get('key') is not tensor:
            memo.update(key=tensor, view=encode(tensor))
        return memo['view']
    teacher.image = image
    pointer = ridge_pointer(device)
    models = fixed_translators(device, MODELS)
    stores = {}
    table, videos = {}, {}
    started = time.monotonic()
    for group, split, dataset, limit in GROUPS:
        store = stores.setdefault(split, Store(split))
        index = {c['case_id']: k for k, c in enumerate(store.cases)}
        rows = select_cases(split, 1, limit, SEED, dataset)
        picks = [(index[r['case_id']], OFFSETS[k % len(OFFSETS)]) for k, r in enumerate(rows)]
        table[group] = {m: {s: {f: [] for f in FIELDS} for s in ('cond', 'alone')} for m in MODELS}
        with ThreadPoolExecutor(4) as pool:
            for start in range(0, len(picks), 8):
                for item in pool.map(lambda job: store.item(*job), picks[start:start + 8]):
                    if item is None:
                        continue
                    videos.setdefault(group, []).append(store.cases[item['case']]['video_id'])
                    for name, translate in models.items():
                        for scenario in ('cond', 'alone'):
                            terms, info = one(teacher, translate, item, scenario, device, pointer)
                            cell = table[group][name][scenario]
                            for field in FIELDS:
                                cell[field].append(float(terms[field] if field in terms else info[field]))
        print(group, len(table[group][MODELS[0]]['cond']['iou']), f'items, {time.monotonic() - started:.0f}s', flush=True)
    (ROOT / 'runs/analysis/gap_proxy.json').write_text(json.dumps(dict(table=table, videos=videos)))
    report(table)
    print('GAP_DONE', flush=True)


if __name__ == '__main__':
    main()
