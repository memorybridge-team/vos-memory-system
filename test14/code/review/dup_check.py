#!/usr/bin/env python3
"""Near-duplicate clips across groups, from thumbnails alone. No model is used.

Video ids keep fit, dev and val apart, but two ids could still be cuts of one source clip. Three frames of every video become 32x32
gray images, centered and scaled to unit length, so the dot product of two frames is their correlation. A video's score is its highest
correlation with any frame of any fit video other than itself. fit against itself shows how alike the training pool already is.
Writes runs/analysis/dup_check.json.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from jf_eval import DATA, ROOT, video_paths

GROUPS = {'fit MOSE': ('fit', 'MOSEv2'), 'fit LVOS': ('fit', 'LVOS v2'), 'dev MOSE': ('dev', 'MOSEv2'), 'dev LVOS': ('dev', 'LVOS v2'), 'val LVOS': ('val', 'LVOS v2')}
LEVELS = (.9, .95, .98, .99)


def thumbnail(path):
    with Image.open(path) as image:
        image.draft('L', (128, 128))
        a = np.asarray(image.convert('L').resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel().copy()
    a -= a.mean()
    norm = np.linalg.norm(a)
    return a / norm if norm > 1 else a * 0


def three(paths):
    return np.stack([thumbnail(paths[i]) for i in (0, len(paths) // 2, len(paths) - 1)])


def listing():
    rows = []
    metas = {name: torch.load(ROOT / 'data' / name / 'meta.pt', map_location='cpu', weights_only=False) for name in ('fit', 'dev', 'val')}
    for group, (split, dataset) in GROUPS.items():
        names = sorted({c['video_id'] for c in metas[split]['cases'] if c['dataset'] == dataset})
        rows += [(group, name, lambda d=dataset, n=name: video_paths(d, n)[0]) for name in names]
    m3 = DATA / 'M3VOS-manual/data'
    names = [x.strip() for x in (m3 / 'ImageSets/val.txt').read_text().splitlines() if x.strip()]
    rows += [('M3-VOS', n, lambda n=n: sorted(p for p in (m3 / 'JPEGImages' / n).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))) for n in names]
    pu = DATA / 'PUMaVOS/extracted/PUBLIC_PUMaVOS'
    names = sorted(p.name for p in (pu / 'JPEGImages').iterdir() if p.is_dir())
    rows += [('PUMaVOS', n, lambda n=n: sorted(p for p in (pu / 'JPEGImages' / n).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))) for n in names]
    return rows


def main():
    torch.set_num_threads(6)
    rows = listing()
    print(len(rows), 'videos', flush=True)
    with ThreadPoolExecutor(8) as pool:
        stacks = list(pool.map(lambda row: three(row[2]()), rows))
    all_frames = torch.from_numpy(np.stack(stacks))  # [videos, 3, 1024]
    group = np.array([r[0] for r in rows])
    name = np.array([r[1] for r in rows])
    fit = np.flatnonzero(np.char.startswith(group, 'fit'))
    bank = all_frames[fit].reshape(-1, 1024)
    summary = {}
    print(f"{'group':10s} {'videos':>6s} {'median':>7s} {'p90':>7s} {'p99':>7s} " + ' '.join(f'>={x:<5}' for x in LEVELS))
    for label in dict.fromkeys(group):
        members = np.flatnonzero(group == label)
        best, match = [], []
        for start in range(0, len(members), 256):
            chunk = members[start:start + 256]
            scores = (all_frames[chunk].reshape(-1, 1024) @ bank.T).reshape(len(chunk), 3, len(fit), 3).amax((1, 3))
            for row, video in enumerate(chunk):
                scores[row, fit == video] = -1
            value, where = scores.max(1)
            best += value.tolist()
            match += [str(name[fit[k]]) for k in where.tolist()]
        best = np.array(best)
        counts = [int((best >= x).sum()) for x in LEVELS]
        print(f'{label:10s} {len(best):6d} {np.median(best):7.3f} {np.percentile(best, 90):7.3f} {np.percentile(best, 99):7.3f} ' + ' '.join(f'{c:<6d}' for c in counts), flush=True)
        order = np.argsort(-best)
        print('   top ' + ', '.join(f'{name[members[k]]}~{match[k]} {best[k]:.3f}' for k in order[:4]), flush=True)
        summary[label] = {str(name[v]): dict(score=float(b), match=m) for v, b, m in zip(members, best, match)}
    (ROOT / 'runs/analysis/dup_check.json').write_text(json.dumps(summary))
    print('DUP_DONE', flush=True)


if __name__ == '__main__':
    main()
