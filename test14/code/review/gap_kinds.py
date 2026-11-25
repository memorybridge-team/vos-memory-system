#!/usr/bin/env python3
"""What kind of frame carries the J&F gap to the Base+ state. Uses stored per-frame scores and predicted areas; nothing is rerun.

Each visible frame after the switch falls in one group, by what the method did where the reference did well or badly:
  gone     method predicts nothing
  wrong    method predicts something with J < .1
  partial  J in .1 .. .7
  good     J >= .7
A case counts as `lost` when its method stays gone or wrong for the last half of its visible frames.
usage: gap_kinds.py <method>@<tag> [<method>@<tag> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/cmmt-design-review-20261104/runs/jf')
REF = ('target_ref', 't2_mose')


def load(tag):
    return {p.name: json.loads(p.read_text()) for p in (ROOT / tag / 'cases').glob('*.json')}


def frames(row, name):
    s = row['scores'][name]
    keep = [i for i, (o, j, v) in enumerate(zip(row['offsets'], s['J'], s['visible'])) if o >= 1 and j is not None and v]
    return np.array([row['offsets'][i] for i in keep]), np.array([s['J'][i] for i in keep]), np.array([s['F'][i] for i in keep]), np.array([s['area'][i] for i in keep])


def kind(j, area):
    k = np.full(j.shape, 'partial', dtype=object)
    k[j >= .7] = 'good'
    k[j < .1] = 'wrong'
    k[area == 0] = 'gone'
    return k


cache = {}
for key in sys.argv[1:] + [f'{REF[0]}@{REF[1]}']:
    tag = key.split('@')[1]
    cache.setdefault(tag, load(tag))
base = cache[REF[1]]
NAMES = ('good', 'partial', 'wrong', 'gone')
for key in sys.argv[1:]:
    method, tag = key.split('@')
    table = {(a, b): 0. for a in NAMES for b in NAMES}
    count = {(a, b): 0 for a in NAMES for b in NAMES}
    per, lost_m, lost_r, cases = [], 0, 0, 0
    for name in sorted(set(cache[tag]) & set(base)):
        row, ref = cache[tag][name], base[name]
        if method not in row['scores'] or REF[0] not in ref['scores']:
            continue
        o, j, f, a = frames(row, method)
        o2, j2, f2, a2 = frames(ref, REF[0])
        if len(o) == 0 or len(o) != len(o2):
            continue
        cases += 1
        kinds, kinds2 = kind(j, a), kind(j2, a2)
        diff = ((j + f) / 2 - (j2 + f2) / 2) * 100 / len(o)
        for x, y, d in zip(kinds, kinds2, diff):
            table[(x, y)] += d
            count[(x, y)] += 1
        tail = len(o) // 2
        lost_m += float(np.isin(kinds[tail:], ('gone', 'wrong')).mean()) > .9
        lost_r += float(np.isin(kinds2[tail:], ('gone', 'wrong')).mean()) > .9
    print(f'{method} ({tag}): {cases} cases. share of the mean gap by (method state, Base+ state), in J&F points')
    print('             Base+: ' + ' '.join(f'{n:>8s}' for n in NAMES))
    for a in NAMES:
        print(f'  method {a:8s}  ' + ' '.join(f'{table[(a, b)] / cases:+8.2f}' for b in NAMES))
    print(f'  frames {sum(count.values())}, share   ' + ' '.join(f'{sum(count[(a, b)] for a in NAMES) / max(sum(count.values()), 1):8.1%}' for b in NAMES) + '  (Base+ state)')
    print(f'  method state share   ' + ' '.join(f'{sum(count[(a, b)] for b in NAMES) / max(sum(count.values()), 1):8.1%}' for a in NAMES))
    print(f'  cases lost in the last half of their frames: method {lost_m}, Base+ {lost_r}')
    print(f'  total {sum(table.values()) / cases:+.2f}')
