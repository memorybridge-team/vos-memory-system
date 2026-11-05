#!/usr/bin/env python3
"""Which evaluation videos show a scene that is also in the training videos? Denser than dup_check, which sampled only three frames.

Twelve evenly spaced frames per video. Frames with almost no contrast are dropped, so fades and black frames cannot create a match.
share = part of a query's usable frames whose best correlation with any fit frame reaches the level. M3-VOS and PUMaVOS are the control:
they come from other sources, so a level at which they match shows the level is too loose. Writes runs/analysis/dup_scan.json.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from dup_check import listing
from jf_eval import ROOT

COUNT = 12
FLAT = 12
LEVELS = (.93, .95, .97, .98)


def load(path):
    with Image.open(path) as image:
        image.draft('L', (128, 128))
        a = np.asarray(image.convert('L').resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel().copy()
    spread = float(a.std())
    a -= a.mean()
    norm = np.linalg.norm(a)
    return (a / norm if norm > 1 else a * 0), spread


def frames(paths):
    rows = [load(paths[int(round(k * (len(paths) - 1) / (COUNT - 1)))]) for k in range(COUNT)]
    return np.stack([r[0] for r in rows]), np.array([r[1] for r in rows])


def main():
    torch.set_num_threads(6)
    rows = listing()
    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(lambda row: frames(row[2]()), rows))
    print('read', len(got), flush=True)
    stack = torch.from_numpy(np.stack([g[0] for g in got]))  # [videos, 12, 1024]
    spread = np.stack([g[1] for g in got])
    group = np.array([r[0] for r in rows])
    name = np.array([r[1] for r in rows])
    fit = np.flatnonzero(np.char.startswith(group, 'fit'))
    keep = spread[fit].ravel() >= FLAT
    bank = stack[fit].reshape(-1, 1024)[torch.from_numpy(keep)]
    owner = np.repeat(fit, COUNT)[keep]
    result = {}
    print(f"{'group':10s} {'videos':>6s}  " + '  '.join(f'at {x}: >=.25 >=.5 >=.75' for x in LEVELS))
    for label in ('dev MOSE', 'dev LVOS', 'val LVOS', 'M3-VOS', 'PUMaVOS'):
        members = np.flatnonzero(group == label)
        table = {}
        for start in range(0, len(members), 128):
            chunk = members[start:start + 128]
            value, where = (stack[chunk].reshape(-1, 1024) @ bank.T).max(1)
            value, where = value.reshape(len(chunk), COUNT).numpy(), where.reshape(len(chunk), COUNT).numpy()
            for row, video in enumerate(chunk):
                usable = spread[video] >= FLAT
                if not usable.any():
                    continue
                top = value[row][usable]
                votes = owner[where[row][usable]]
                best = np.bincount(votes).argmax()
                table[str(name[video])] = dict(shares={str(x): float((top >= x).mean()) for x in LEVELS}, peak=float(top.max()),
                                               match=str(name[best]), match_group=str(group[best]), usable=int(usable.sum()))
        result[label] = table
        cells = []
        for x in LEVELS:
            shares = np.array([v['shares'][str(x)] for v in table.values()])
            cells.append(f'{int((shares >= .25).sum()):8d} {int((shares >= .5).sum()):4d} {int((shares >= .75).sum()):4d}')
        print(f'{label:10s} {len(table):6d}  ' + '  '.join(f'        {c}' for c in cells), flush=True)
    (ROOT / 'runs/analysis/dup_scan.json').write_text(json.dumps(result))
    print('SCAN_DONE', flush=True)


if __name__ == '__main__':
    main()
