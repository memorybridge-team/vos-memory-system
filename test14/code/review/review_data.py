"""Flat record arrays for the design review."""
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ['REVIEW_ROOT'])
SCALES = json.loads((ROOT / 'data/normalization.json').read_text())['scales']


class Split:
    def __init__(self, name):
        folder = ROOT / 'data' / name
        self.name = name
        self.meta = torch.load(folder / 'meta.pt', map_location='cpu', weights_only=False)
        n = self.n = self.meta['count']
        self.src_sp = np.memmap(folder / 'src_sp.i16', dtype=np.int16, mode='r', shape=(n, 64, 64, 64))
        self.tgt_sp = np.memmap(folder / 'tgt_sp.i16', dtype=np.int16, mode='r', shape=(n, 64, 64, 64))
        self.src_p = np.memmap(folder / 'src_p.f32', dtype=np.float32, mode='r', shape=(n, 256))
        self.tgt_p = np.memmap(folder / 'tgt_p.f32', dtype=np.float32, mode='r', shape=(n, 256))
        cases = self.meta['cases']
        self.is_lvos = torch.tensor([c['dataset'] != 'MOSEv2' for c in cases])[self.meta['case_id'].long()]

    def spatial(self, index, device):
        """Records as float32 tensors [N,64,64,64] on the device."""
        a = torch.from_numpy(np.array(self.src_sp[index])).to(device, non_blocking=True)
        b = torch.from_numpy(np.array(self.tgt_sp[index])).to(device, non_blocking=True)
        return a.view(torch.bfloat16).float(), b.view(torch.bfloat16).float()

    def pointers(self, device=None):
        a = torch.from_numpy(np.array(self.src_p))
        b = torch.from_numpy(np.array(self.tgt_p))
        return (a, b) if device is None else (a.to(device), b.to(device))


def chunks(n, size):
    for start in range(0, n, size):
        yield np.arange(start, min(start + size, n))


class Moments:
    """FP64 sufficient statistics on the device."""

    def __init__(self, dx, dy, device):
        self.n = 0
        self.sx = torch.zeros(dx, dtype=torch.float64, device=device)
        self.sy = torch.zeros(dy, dtype=torch.float64, device=device)
        self.xx = torch.zeros(dx, dx, dtype=torch.float64, device=device)
        self.xy = torch.zeros(dx, dy, dtype=torch.float64, device=device)
        self.yy = torch.zeros(dy, dtype=torch.float64, device=device)

    def update(self, x, y):
        x, y = x.double(), y.double()
        self.n += x.shape[0]
        self.sx += x.sum(0)
        self.sy += y.sum(0)
        self.xx += x.T @ x
        self.xy += x.T @ y
        self.yy += y.square().sum(0)

    def solve(self, alpha):
        mx, my = self.sx / self.n, self.sy / self.n
        g = self.xx / self.n - torch.outer(mx, mx)
        c = self.xy / self.n - torch.outer(mx, my)
        lam = alpha * torch.trace(g) / g.shape[0]
        w = torch.linalg.solve(g + lam * torch.eye(g.shape[0], dtype=g.dtype, device=g.device), c)
        return w, my - mx @ w

    def sse(self, w, b):
        """Exact SSE of x@w+b from the sufficient statistics (this object's data)."""
        a = w.T @ self.xy
        cross = a.diagonal().sum() + b @ self.sy
        quad = (w.T @ self.xx @ w).diagonal().sum() + 2 * (b @ (w.T @ self.sx)) + self.n * b.square().sum()
        return float(self.yy.sum() - 2 * cross + quad)
