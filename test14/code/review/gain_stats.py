#!/usr/bin/env python3
"""How much each model flattens the conditioning record: output contrast against the target's.

Contrast is measured around each record's channel mean, the same center `rescale:` uses.
energy = output variance / target variance; slope = best MSE gain on the output; corr = correlation.
"""
import sys

import numpy as np
import torch

from jf_eval import LEGACY, restore_model
from review_data import ROOT, SCALES, Split
from review_models import load_run

STEP = 6  # every sixth case keeps this quick on CPU


def models(device):
    ridge = torch.load(ROOT / 'runs/ridge/ridge.pt', map_location=device, weights_only=True)
    w, b = ridge['spatial_w'], ridge['spatial_b']
    yield 'ridge', lambda x, p: x.movedim(1, -1).double().matmul(w).add(b).movedim(-1, 1).float()
    raw = torch.load(LEGACY, map_location='cpu', weights_only=True)
    legacy = restore_model('transformer', raw.get('transformer', raw)).to(device).float().eval()
    yield 'legacy', lambda x, p: legacy.translate_tensors(x[None, None], p[None, None])[0][0, 0]
    for name in sys.argv[2:]:
        run = load_run(ROOT / 'runs/spatial' / name, device)
        yield name, lambda x, p, run=run: run(x, p, None)[0]


def main():
    torch.set_num_threads(8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = Split(sys.argv[1])
    cond = split.meta['cond'].numpy().astype(bool)
    owner = split.meta['case_id'].numpy()
    cases = np.unique(owner)[::STEP]
    print(f'{split.name}: {len(cases)} cases')
    print('  model        group         nMSE   energy  slope   corr')
    for name, translate in models(device):
        sums = {'cond': np.zeros(5), 'recent': np.zeros(5)}
        for case in cases:
            keep = np.nonzero(owner == case)[0]
            x, y = split.spatial(keep, device)
            p = torch.from_numpy(np.array(split.src_p[keep])).to(device)
            with torch.inference_mode():
                h = translate(x, p).float()
            a, c = h - h.mean((2, 3), keepdim=True), y - y.mean((2, 3), keepdim=True)
            flag = torch.from_numpy(cond[keep]).to(device)
            for group, mask in (('cond', flag), ('recent', ~flag)):
                if mask.any():
                    sums[group] += [float((h - y)[mask].square().sum()), float(a[mask].square().sum()), float(c[mask].square().sum()),
                                    float((a[mask] * c[mask]).sum()), int(mask.sum()) * 64 ** 3]
        for group, v in sums.items():
            print(f'  {name:12s} {group:8s} {v[0] / v[4] / SCALES["spatial"] ** 2:6.4f}  {v[1] / v[2]:6.4f}  {v[3] / v[1]:6.4f}  {v[3] / np.sqrt(v[1] * v[2]):6.4f}', flush=True)


if __name__ == '__main__':
    main()
