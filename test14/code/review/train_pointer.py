#!/usr/bin/env python3
"""Pointer design experiments. All pointer data fits on the GPU, so each model takes minutes."""
import json
import math
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from review_data import ROOT, SCALES, Moments, Split

device = torch.device('cuda')
OUT = ROOT / 'runs/pointer'
OUT.mkdir(parents=True, exist_ok=True)
SP = SCALES['pointer'] ** 2


def load(name):
    split = Split(name)
    m = split.meta
    x, y = split.pointers(device)
    cases = m['case_id'].long().to(device)
    cond = m['cond'].to(device)
    sa, ta = (m['src_pres'] <= 0).to(device), (m['tgt_pres'] <= 0).to(device)
    n = split.n
    same_prev = torch.zeros(n, dtype=torch.bool, device=device)
    same_next = torch.zeros(n, dtype=torch.bool, device=device)
    same_prev[1:] = cases[1:] == cases[:-1]
    same_next[:-1] = cases[1:] == cases[:-1]
    # Neighbour source pointers in the same stored state; never the target.
    prev_p, next_p = x.clone(), x.clone()
    prev_p[1:][same_prev[1:]] = x[:-1][same_prev[1:]]
    next_p[:-1][same_next[:-1]] = x[1:][same_next[:-1]]
    # Mean of present, non-conditioning source pointers in the case.
    use = (~sa & ~cond).double()
    count = torch.zeros(int(cases.max()) + 1, dtype=torch.float64, device=device).index_add_(0, cases, use)
    total = torch.zeros(int(cases.max()) + 1, 256, dtype=torch.float64, device=device).index_add_(0, cases, x.double() * use[:, None])
    case_mean = (total / count.clamp_min(1)[:, None]).float()[cases]
    case_some = (count > 0)[cases]
    pool = torch.load(ROOT / 'data' / name / 'src_pool8.pt', map_location='cpu', weights_only=True)
    return dict(name=name, x=x, y=y, cond=cond, sa=sa, ta=ta, prev=prev_p, next=next_p, mean=case_mean, some=case_some,
                pool=pool, lvos=split.is_lvos.to(device), n=n)


def extra_features(d, kind, index=None):
    i = slice(None) if index is None else index
    parts = []
    flags = torch.stack((d['cond'][i], d['sa'][i]), -1).float()
    if 'flags' in kind:
        parts.append(flags)
    if 'temporal' in kind:
        parts += [d['prev'][i], d['next'][i], d['mean'][i], d['some'][i, None].float()]
    if 'smean' in kind:
        cpu = index.cpu() if index is not None else slice(None)
        parts.append(d['pool'][cpu].mean((2, 3)).to(device))
    return torch.cat(parts, -1) if parts else None


class Net(nn.Module):
    def __init__(self, ridge, kind, hidden=512, depth=1, norm=False, extra=0, freeze=False, use_pool=False):
        super().__init__()
        self.skip = nn.Linear(256, 256)
        with torch.no_grad():
            self.skip.weight.copy_(ridge[0].T.float())
            self.skip.bias.copy_(ridge[1].float())
        self.skip.requires_grad_(not freeze)
        size = 256 + extra
        self.use_pool = use_pool
        if use_pool:
            self.proj = nn.Linear(64, 128)
            self.pos = nn.Parameter(torch.randn(1, 64, 128) * 0.02)
            self.query = nn.Linear(256, 128)
            self.attn = nn.MultiheadAttention(128, 4, batch_first=True)
            size += 128
        layers = []
        for i in range(depth):
            layers += [nn.Linear(size if i == 0 else hidden, hidden), nn.GELU()]
            if norm:
                layers.append(nn.LayerNorm(hidden))
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, 256)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, p, extra, pool):
        parts = [p]
        if extra is not None:
            parts.append(extra)
        if self.use_pool:
            tokens = self.proj(pool.flatten(2).transpose(1, 2)) + self.pos
            q = self.query(p)[:, None]
            parts.append(self.attn(q, tokens, tokens, need_weights=False)[0][:, 0])
        return self.skip(p) + self.out(self.body(torch.cat(parts, -1)))


def report(name, pred, d):
    e = (pred - d['y']).square().mean(1) / SP
    mask = {'all': torch.ones_like(e, dtype=torch.bool), 'cond': d['cond'], 'both': ~d['sa'] & ~d['ta'] & ~d['cond'],
            'neither': d['sa'] & d['ta'], 'src_only': ~d['sa'] & d['ta'], 'tgt_only': d['sa'] & ~d['ta']}
    row = {k: float(e[v].mean()) for k, v in mask.items() if bool(v.any())}
    if bool(d['lvos'].any()) and bool((~d['lvos']).any()):
        row['lvos'] = float(e[d['lvos']].mean())
        row['mose'] = float(e[~d['lvos']].mean())
    row['energy'] = float(pred.square().mean() / d['y'].square().mean())
    return row


@torch.inference_mode()
def predict(model, d, kind):
    outputs = []
    for start in range(0, d['n'], 4096):
        i = torch.arange(start, min(start + 4096, d['n']), device=device)
        pool = d['pool'][i.cpu()].to(device) if model.use_pool else None
        outputs.append(model(d['x'][i], extra_features(d, kind, i), pool))
    return torch.cat(outputs)


def train(name, fit, dev, val, ridge, kind='', updates=12000, batch=1024, lr=1e-3, seed=7, weight_decay=1e-4, mask=None, **options):
    path = OUT / f'{name}.json'
    if path.exists():
        print('exists', name)
        return json.loads(path.read_text())
    torch.manual_seed(seed)
    use_pool = 'pool' in kind and 'smean' not in kind
    probe = extra_features(fit, kind, torch.arange(4, device=device))
    extra = 0 if probe is None else probe.shape[-1]
    model = Net(ridge, kind, extra=extra, use_pool=use_pool, **options).to(device)
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim > 1 and 'pos' not in n]
    other = [p for n, p in model.named_parameters() if p.requires_grad and not (p.ndim > 1 and 'pos' not in n)]
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=weight_decay), dict(params=other, weight_decay=0.)], lr=lr)
    candidates = torch.arange(fit['n'], device=device) if mask is None else mask.nonzero().flatten()
    g = torch.Generator(device=device).manual_seed(seed)
    started = time.monotonic()
    for step in range(updates):
        rate = lr * min(1, (step + 1) / 500) * (0.5 * (1 + math.cos(math.pi * step / updates)) * 0.99 + 0.01)
        for group in optimizer.param_groups:
            group['lr'] = rate
        i = candidates[torch.randint(len(candidates), (batch,), generator=g, device=device)]
        pool = fit['pool'][i.cpu()].to(device) if use_pool else None
        loss = (model(fit['x'][i], extra_features(fit, kind, i), pool) - fit['y'][i]).square().mean() / SP
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    row = dict(name=name, kind=kind, parameters=sum(p.numel() for p in model.parameters()),
               fit=report(name, predict(model, fit, kind), fit), dev=report(name, predict(model, dev, kind), dev),
               val=report(name, predict(model, val, kind), val), seconds=round(time.monotonic() - started, 1), options=str(options))
    path.write_text(json.dumps(row, indent=1))
    torch.save(model.state_dict(), OUT / f'{name}.pt')
    print(json.dumps({k: row[k] for k in ('name', 'parameters', 'val', 'dev', 'seconds')}), flush=True)
    return row


def closed_form(fit, dev, val):
    results = {}
    def solve(mask):
        m = Moments(256, 256, device)
        m.update(fit['x'][mask], fit['y'][mask])
        return m
    base = solve(torch.ones(fit['n'], dtype=torch.bool, device=device))
    alpha = 1e-6
    w, b = base.solve(alpha)
    apply = lambda x, w, b: (x.double() @ w + b).float()
    results['ridge'] = {s: report('ridge', apply(d['x'], w, b), d) for s, d in (('fit', fit), ('dev', dev), ('val', val))}
    # Exact rule: a source no-object pointer carries no information, so use the best constant.
    keep = ~fit['sa']
    pm = solve(keep)
    w2, b2 = pm.solve(alpha)
    const = fit['y'][fit['sa']].mean(0)
    def ruled(d):
        p = apply(d['x'], w2, b2)
        p[d['sa']] = const
        return p
    results['ridge_rule'] = {s: report('ridge_rule', ruled(d), d) for s, d in (('fit', fit), ('dev', dev), ('val', val))}
    # Separate conditioning map as well.
    keep_rest = ~fit['sa'] & ~fit['cond']
    keep_cond = ~fit['sa'] & fit['cond']
    wr, br = solve(keep_rest).solve(alpha)
    wc, bc = solve(keep_cond).solve(1e-2)
    def split(d):
        p = apply(d['x'], wr, br)
        c = d['cond'] & ~d['sa']
        p[c] = apply(d['x'][c], wc, bc)
        p[d['sa']] = const
        return p
    results['ridge_rule_cond'] = {s: report('ridge_rule_cond', split(d), d) for s, d in (('fit', fit), ('dev', dev), ('val', val))}
    (OUT / 'closed_form.json').write_text(json.dumps(results, indent=1))
    for k, v in results.items():
        print(k, json.dumps({a: b for a, b in v['val'].items()}), 'dev', round(v['dev']['all'], 5), flush=True)
    return (w.float(), b.float()), (wr.float(), br.float())


def main():
    fit, dev, val = load('fit'), load('dev'), load('val')
    print('loaded', flush=True)
    ridge, _ = closed_form(fit, dev, val)
    common = dict(fit=fit, dev=dev, val=val, ridge=ridge)
    present = ~fit['sa']
    train('mlp_current_b64', **common, updates=30000, batch=64, lr=3e-4)
    train('mlp', **common)
    train('mlp_frozen_skip', **common, freeze=True)
    train('mlp_wide2048', **common, hidden=2048)
    train('mlp_deep3', **common, hidden=1024, depth=3, norm=True)
    train('flags', **common, kind='flags')
    train('temporal', **common, kind='temporal')
    train('temporal_flags', **common, kind='temporal flags')
    train('smean', **common, kind='smean')
    train('spool', **common, kind='pool')
    train('all_big', **common, kind='temporal flags pool', hidden=1024, depth=3, norm=True, updates=20000)


if __name__ == '__main__':
    main()
