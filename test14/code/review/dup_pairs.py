#!/usr/bin/env python3
"""Are the flagged pairs one clip, or one blank frame? Eight evenly spaced frames of each side of a flagged pair are compared.

overlap = share of the query's frames that have a frame with correlation >= .98 in the matched video. Frames with almost no contrast
do not count, so fades and black frames cannot make a pair look shared. Writes runs/analysis/dup_pairs.json and a contact sheet.
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from jf_eval import ROOT, video_paths
from dup_check import listing

COUNT = 8
LEVEL = .98
FLAT = 12  # gray-level standard deviation below which a frame shows nothing


def load(path):
    with Image.open(path) as image:
        image.draft('L', (128, 128))
        a = np.asarray(image.convert('L').resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel().copy()
    spread = float(a.std())
    a -= a.mean()
    norm = np.linalg.norm(a)
    return (a / norm if norm > 1 else a * 0), spread


def spaced(paths):
    return [paths[int(round(k * (len(paths) - 1) / (COUNT - 1)))] for k in range(COUNT)]


def frames(paths):
    rows = [load(p) for p in spaced(paths)]
    return np.stack([r[0] for r in rows]), np.array([r[1] for r in rows])


def overlap(a, b):
    x, sx = frames(a)
    y, sy = frames(b)
    sim = x @ y.T
    sim[sx < FLAT, :] = -1
    sim[:, sy < FLAT] = -1
    useful = int((sx >= FLAT).sum())
    return float((sim.max(1) >= LEVEL).sum() / max(useful, 1)), useful, float(sim.max())


def sheet(pairs, paths, out):
    width = 96
    canvas = Image.new('RGB', (width * 6 + 20, (width * 9 // 16 + 6) * len(pairs)), 'white')
    for row, (q, m) in enumerate(pairs):
        for side, name in enumerate((q, m)):
            chosen = [paths[name][k] for k in (0, len(paths[name]) // 2, len(paths[name]) - 1)]
            for col, path in enumerate(chosen):
                with Image.open(path) as image:
                    thumb = image.convert('RGB').resize((width, width * 9 // 16), Image.BILINEAR)
                canvas.paste(thumb, (col * width + side * (width * 3 + 20), row * (width * 9 // 16 + 6)))
    canvas.save(out)


def main():
    rows = listing()
    lookup = {name: fn for _, name, fn in rows}
    summary = json.loads((ROOT / 'runs/analysis/dup_check.json').read_text())
    result, samples = {}, {}
    for group in ('dev MOSE', 'dev LVOS', 'val LVOS', 'M3-VOS', 'PUMaVOS'):
        flagged = [(n, v['match'], v['score']) for n, v in summary[group].items() if v['score'] >= .95]
        with ThreadPoolExecutor(8) as pool:
            got = list(pool.map(lambda r: overlap(lookup[r[0]](), lookup[r[1]]()), flagged))
        table = {n: dict(match=m, score=s, overlap=g[0], useful=g[1], peak=g[2]) for (n, m, s), g in zip(flagged, got)}
        result[group] = table
        shares = np.array([v['overlap'] for v in table.values()]) if table else np.zeros(0)
        print(f"{group:9s} videos {len(summary[group])} flagged {len(table)} blank-only {int(sum(v['peak'] < LEVEL for v in table.values()))} "
              f'overlap>=.25 {int((shares >= .25).sum())} >=.5 {int((shares >= .5).sum())} >=.75 {int((shares >= .75).sum())} =1 {int((shares >= 1).sum())}', flush=True)
        ranked = sorted(table.items(), key=lambda kv: -kv[1]['overlap'])
        random.Random(7).shuffle(ranked)
        samples[group] = [(n, v['match']) for n, v in ranked if v['overlap'] >= .25][:6]
    (ROOT / 'runs/analysis/dup_pairs.json').write_text(json.dumps(result))
    pairs = samples['dev MOSE'][:6] + samples['val LVOS'][:5] + samples['dev LVOS'][:2]
    paths = {n: lookup[n]() for pair in pairs for n in pair}
    sheet(pairs, paths, ROOT / 'runs/analysis/dup_sheet.png')
    print('sheet rows', pairs, flush=True)
    print('PAIRS_DONE', flush=True)


if __name__ == '__main__':
    main()
