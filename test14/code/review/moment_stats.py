#!/usr/bin/env python3
"""Moment-Matched Copy statistics from the fit split (doc 01 section 3.2).

Spatial memory per channel and object pointer per dimension, separately for conditioning and
non-conditioning records, valid records only. Fit videos only. Writes runs/moment/moment.pt.
"""
import time

import numpy as np
import torch

from review_data import ROOT, Split, chunks

SIZE = 2048


def spatial_moments(split, array, mask):
    """(mean, std) per channel of the selected records."""
    total, square, count = torch.zeros(64, dtype=torch.float64), torch.zeros(64, dtype=torch.float64), 0
    for index in chunks(split.n, SIZE):
        keep = index[mask[index]]
        if not len(keep):
            continue
        x = torch.from_numpy(np.array(array[keep])).view(torch.bfloat16).float()
        total += x.sum((0, 2, 3), dtype=torch.float64)
        square += x.square().sum((0, 2, 3), dtype=torch.float64)
        count += len(keep) * 64 * 64
    mean = total / count
    return mean, (square / count - mean.square()).clamp_min(0).sqrt()


def main():
    started = time.monotonic()
    torch.set_num_threads(8)
    split = Split('fit')
    cond = split.meta['cond'].numpy().astype(bool)
    out = dict(spatial={}, pointer={}, counts=dict(cond=int(cond.sum()), rest=int((~cond).sum())))
    source_pointer, target_pointer = split.pointers()
    for group, mask in (('cond', cond), ('rest', ~cond)):
        ms, ss = spatial_moments(split, split.src_sp, mask)
        print(group, 'source spatial done', round(time.monotonic() - started), flush=True)
        mt, st = spatial_moments(split, split.tgt_sp, mask)
        print(group, 'target spatial done', round(time.monotonic() - started), flush=True)
        out['spatial'][group] = dict(ms=ms, ss=ss, mt=mt, st=st)
        keep = torch.from_numpy(mask)
        a, b = source_pointer[keep].double(), target_pointer[keep].double()
        out['pointer'][group] = dict(ms=a.mean(0), ss=a.std(0), mt=b.mean(0), st=b.std(0))
    folder = ROOT / 'runs/moment'
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(out, folder / 'moment.pt')
    print('saved', out['counts'], round(time.monotonic() - started), flush=True)


if __name__ == '__main__':
    main()
