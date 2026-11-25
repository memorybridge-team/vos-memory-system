#!/usr/bin/env python3
"""Writes runs/spatial/<tag>_last/model.pt from the last update of a fine-tuning run.

model.pt of such a run holds the best validation by IoU on a few hundred items, where one item moves the score by .004,
so it can freeze on an early step. The feature error falls smoothly to the end and the learning rate ends near zero.
"""
import sys
from pathlib import Path

import torch

ROOT = Path('/workspace/cmmt-design-review-20261104')

for tag in sys.argv[1:]:
    folder = ROOT / 'runs/spatial' / tag
    name = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)['name']
    last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
    out = ROOT / 'runs/spatial' / f'{tag}_last'
    out.mkdir(exist_ok=True)
    torch.save(dict(name=name, state=last['state'], step=last['step']), out / 'model.pt')
    print(tag, name, 'step', last['step'], 'written to', out, flush=True)
