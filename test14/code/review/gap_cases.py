#!/usr/bin/env python3
"""How the J&F gap of a method to the Base+ state is spread across cases. Uses visible frames after the switch only, as paired_acc.py does.

usage: gap_cases.py <method> <tag holding method> [reference method] [tag holding reference]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/cmmt-design-review-20261104/runs/jf')
method, tag = sys.argv[1], sys.argv[2]
ref = sys.argv[3] if len(sys.argv) > 3 else 'target_ref'
ref_tag = sys.argv[4] if len(sys.argv) > 4 else 't2_mose'


def load(t):
    return {p.name: json.loads(p.read_text()) for p in (ROOT / t / 'cases').glob('*.json')}


def frames(row, name):
    s = row['scores'][name]
    return {o: (j + f) / 2 for o, j, f, v in zip(row['offsets'], s['J'], s['F'], s['visible']) if o >= 1 and j is not None and v}


a, b = load(tag), load(ref_tag)
rows = []
for k in sorted(set(a) & set(b)):
    if method not in a[k]['scores'] or ref not in b[k]['scores']:
        continue
    x, y = frames(a[k], method), frames(b[k], ref)
    if not x or not y:
        continue
    area = np.mean([v for v, w in zip(b[k]['scores'][ref]['area'], b[k]['scores'][ref]['visible']) if w] or [0])
    c = a[k]['case']
    first = next((o for o in sorted(x) if o in y and abs(x[o] - y[o]) > .2), None)
    rows.append(dict(key=k, d=100 * (np.mean(list(x.values())) - np.mean(list(y.values()))), base=100 * np.mean(list(y.values())), n=len(y), area=float(area),
                     first=first, name=f"{c.get('video', '?')}/{c.get('object', c.get('obj', '?'))}", switch=a[k].get('switch')))
d = np.array([r['d'] for r in rows])
print(f'{len(rows)} cases, mean {d.mean():+.3f}, median {np.median(d):+.3f}')
print('within 0.5:', int((abs(d) < .5).sum()), ' within 2:', int((abs(d) < 2).sum()), ' below -2:', int((d < -2).sum()), ' above +2:', int((d > 2).sum()),
      ' below -10:', int((d < -10).sum()), ' above +10:', int((d > 10).sum()), ' below -30:', int((d < -30).sum()), ' above +30:', int((d > 30).sum()))
order = np.argsort(d)
loss, gain = -d[d < 0].sum(), d[d > 0].sum()
print(f'total loss {loss:.0f}, total gain {gain:.0f}, net {gain - loss:+.0f} over {len(d)} cases')
for top in (3, 5, 10, 20, 40):
    print(f'  worst {top:2d} cases carry {-d[order[:top]].sum():5.0f} ({-d[order[:top]].sum() / max(loss, 1e-9):4.0%}) of the loss')
print('worst 14 (diff, Base+ score, frames, mean area, first differing offset):')
for i in order[:14]:
    r = rows[i]
    print(f"  {r['d']:+7.2f}  base+ {r['base']:6.2f}  frames {r['n']:4d}  area {r['area']:.5f}  first {r['first']}  {r['name']}")
print('best 8:')
for i in order[::-1][:8]:
    r = rows[i]
    print(f"  {r['d']:+7.2f}  base+ {r['base']:6.2f}  frames {r['n']:4d}  area {r['area']:.5f}  first {r['first']}  {r['name']}")
small = np.array([r['area'] for r in rows])
for lo, hi in ((0, .001), (.001, .01), (.01, .05), (.05, 1.1)):
    m = (small >= lo) & (small < hi)
    if m.any():
        print(f'  area {lo:.3f}..{hi:.3f}: n {int(m.sum()):3d}  mean diff {d[m].mean():+6.2f}  base+ {np.mean([rows[i]["base"] for i in np.nonzero(m)[0]]):6.2f}')
