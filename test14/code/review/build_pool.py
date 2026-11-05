#!/usr/bin/env python3
"""Average-pooled source memory (8x8) for pointer experiments that read spatial context."""
import numpy as np
import torch
from torch.nn import functional as F

from review_data import ROOT, Split, chunks

device = torch.device('cuda')
for name in ('fit', 'dev', 'val'):
    path = ROOT / 'data' / name / 'src_pool8.pt'
    if path.exists():
        continue
    split = Split(name)
    out = torch.zeros(split.n, 64, 8, 8, dtype=torch.float32)
    for index in chunks(split.n, 512):
        a = torch.from_numpy(np.array(split.src_sp[index])).to(device).view(torch.bfloat16).float()
        out[torch.from_numpy(index)] = F.avg_pool2d(a, 8).cpu()
    torch.save(out, path)
    print(name, 'done', flush=True)
