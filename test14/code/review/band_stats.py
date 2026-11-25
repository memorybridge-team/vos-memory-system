#!/usr/bin/env python3
"""Where each model loses contrast, by record group, spatial band and channel. No J&F is used.

Everything is measured around each record's channel mean, the center `rescale:` uses. On dev records:
  energy  output variance / target variance; 1/sqrt(energy) is the gain that restores the variance
  slope   best MSE gain on the output
  corr    correlation with the target
Bands: low = 8x8 average pooled and upsampled, high = the remainder.
Writes runs/calib/<model>.pt with energy-matching gains per channel, per group and per band.
"""
import sys

import numpy as np
import torch
from torch.nn import functional as F

from jf_eval import LEGACY, restore_model
from review_data import ROOT, SCALES, Split
from review_models import load_run

STEP = 8
NAMES = sys.argv[1:] or ('mlp', 'mlp_context', 'ds3', 'ds_context')


def models(device):
    if not sys.argv[1:]:
        ridge = torch.load(ROOT / 'runs/ridge/ridge.pt', map_location=device, weights_only=True)
        w, b = ridge['spatial_w'], ridge['spatial_b']
        yield 'ridge', lambda x, p: x.movedim(1, -1).double().matmul(w).add(b).movedim(-1, 1).float()
        raw = torch.load(LEGACY, map_location='cpu', weights_only=True)
        legacy = restore_model('transformer', raw.get('transformer', raw)).to(device).float().eval()
        yield 'legacy', lambda x, p: legacy.translate_tensors(x[None, None], p[None, None])[0][0, 0]
    for name in NAMES:
        run = load_run(ROOT / 'runs/spatial' / name, device)
        yield name, lambda x, p, run=run: run(x, p, None)[0]


def split_band(a):
    low = F.interpolate(F.avg_pool2d(a, 8), size=a.shape[-2:], mode='bilinear', align_corners=False)
    return low, a - low


def main():
    torch.set_num_threads(8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = Split('dev')
    print('meta keys', sorted(split.meta.keys()), flush=True)
    cond = split.meta['cond'].numpy().astype(bool)
    owner = split.meta['case_id'].numpy()
    cases = np.unique(owner)[::STEP]
    print(f'{len(cases)} cases on {device}', flush=True)
    folder = ROOT / 'runs/calib'
    folder.mkdir(parents=True, exist_ok=True)
    print('  model        group  band    energy  gain   slope   corr   nMSE')
    for name, translate in models(device):
        sums = {g: {b: torch.zeros(4, 64, dtype=torch.float64) for b in ('all', 'low', 'high')} for g in ('cond', 'rest')}
        counts = dict(cond=0, rest=0)
        for case in cases:
            keep = np.nonzero(owner == case)[0]
            x, y = split.spatial(keep, device)
            p = torch.from_numpy(np.array(split.src_p[keep])).to(device)
            with torch.inference_mode():
                h = translate(x, p).float()
            a, c = h - h.mean((2, 3), keepdim=True), y - y.mean((2, 3), keepdim=True)
            al, ah = split_band(a)
            cl, ch = split_band(c)
            flag = torch.from_numpy(cond[keep]).to(device)
            for group, mask in (('cond', flag), ('rest', ~flag)):
                if not mask.any():
                    continue
                counts[group] += int(mask.sum())
                for band, u, v in (('all', a, c), ('low', al, cl), ('high', ah, ch)):
                    u, v = u[mask], v[mask]
                    row = torch.stack((u.square().sum((0, 2, 3)), v.square().sum((0, 2, 3)), (u * v).sum((0, 2, 3)), (u - v).square().sum((0, 2, 3))))
                    sums[group][band] += row.double().cpu()
        out = {}
        for group, bands in sums.items():
            out[group] = {}
            for band, v in bands.items():
                t = v.sum(1)
                energy, slope, corr = t[0] / t[1], t[2] / t[0], t[2] / (t[0] * t[1]).sqrt()
                nmse = t[3] / (counts[group] * 64 ** 3) / SCALES['spatial'] ** 2
                print(f'  {name:12s} {group:5s}  {band:5s}  {energy:6.4f} {1 / energy.sqrt():6.3f} {slope:6.4f} {corr:6.4f} {nmse:6.4f}', flush=True)
                per = (v[1] / v[0]).sqrt().float()
                out[group][band] = per
                if band == 'all':
                    q = np.percentile(per.numpy(), [0, 10, 50, 90, 100])
                    print(f'  {"":12s} per-channel gain min/10/50/90/max ' + ' '.join(f'{x:.3f}' for x in q), flush=True)
        torch.save(out, folder / f'{name}.pt')


if __name__ == '__main__':
    main()
