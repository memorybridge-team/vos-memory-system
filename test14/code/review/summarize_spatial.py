#!/usr/bin/env python3
"""Spatial candidates ordered by official LVOS validation error."""
import glob
import json
import os

import torch

from review_models import make_spatial

ROOT = '/workspace/cmmt-design-review-20261104/runs/spatial/'
ridge = torch.load('/workspace/cmmt-design-review-20261104/runs/ridge/ridge.pt', map_location='cpu', weights_only=True)
rows = []
for path in glob.glob(ROOT + '*/result.json'):
    data = json.load(open(path))
    data['tag'] = path.split('/')[-2]
    rows.append(data)
print('tag                     params   val_all  val_cond val_recent | dev_all   lvos    mose  | minutes')
print('ridge (closed form)       4160   0.16760  (see analysis)     | 0.16488')
print('legacy (fit+dev trained) 194304  0.14619  0.13865            | biased, trained on dev')
for r in sorted(rows, key=lambda x: x['val']['all']):
    v, d = r['val'], r['dev']
    print(f"{r['tag']:22s} {r['parameters']:8d}  {v['all']:.5f}  {v['cond']:.5f}  {v['recent']:.5f}  | {d['all']:.5f} {d.get('lvos', 0):.5f} {d.get('mose', 0):.5f} | {r['seconds'] / 60:.0f}")
