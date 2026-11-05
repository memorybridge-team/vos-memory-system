#!/usr/bin/env python3
"""Train one spatial candidate on fit records; score on dev and official LVOS validation."""
import argparse
import json
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from review_data import ROOT, SCALES, Split, chunks
from review_models import make_spatial
from vos_memory_inspector.state_training.policy import optimizer_groups


class Reader:
    """Random record batches through threaded direct reads; several times faster than memmap faults."""

    UNIT = 524288

    def __init__(self, name, threads=8, depth=4):
        folder = ROOT / 'data' / name
        self.a = os.open(folder / 'src_sp.i16', os.O_RDONLY)
        self.b = os.open(folder / 'tgt_sp.i16', os.O_RDONLY)
        self.pool = ThreadPoolExecutor(threads)
        self.ahead = ThreadPoolExecutor(1)
        self.depth = depth

    def _one(self, job):
        fd, record, view = job
        os.preadv(fd, [view], int(record) * self.UNIT)

    def read(self, index):
        index = np.sort(index)
        a = np.empty((len(index), 64, 64, 64), dtype=np.int16)
        b = np.empty_like(a)
        jobs = [(self.a, r, memoryview(a[k])) for k, r in enumerate(index)]
        jobs += [(self.b, r, memoryview(b[k])) for k, r in enumerate(index)]
        list(self.pool.map(self._one, jobs))
        return torch.from_numpy(a), torch.from_numpy(b)

    def batches(self, order, batch):
        pending = deque()
        for start in range(0, len(order) - batch + 1, batch):
            pending.append(self.ahead.submit(self.read, order[start:start + batch]))
            if len(pending) >= self.depth:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def decode(a, device):
    return a.to(device, non_blocking=True).view(torch.bfloat16).float()


@torch.inference_mode()
def score(model, split, device, stride=1, keep=False):
    model.eval()
    index = np.arange(0, split.n, stride)
    errors = torch.zeros(len(index), dtype=torch.float64)
    done = 0
    for part in chunks(len(index), 64):
        x, y = split.spatial(index[part], device)
        with torch.autocast('cuda', dtype=torch.float32, enabled=False):
            a = model(x)
        errors[part] = (a - y).square().flatten(1).mean(1).double().cpu()
    model.train()
    nmse = errors / SCALES['spatial'] ** 2
    meta = split.meta
    cond = meta['cond'][index]
    present = (meta['src_pres'][index] > 0)
    age = (meta['switch'] - meta['frame'])[index]
    used = cond | (age <= 5)
    lvos = split.is_lvos[index]
    row = dict(all=float(nmse.mean()), cond=float(nmse[cond].mean()), recent=float(nmse[~cond].mean()),
               used=float(nmse[used].mean()), absent=float(nmse[~present].mean()) if (~present).any() else None)
    if lvos.any() and (~lvos).any():
        row.update(lvos=float(nmse[lvos].mean()), mose=float(nmse[~lvos].mean()))
    return (row, nmse) if keep else row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--updates', type=int, default=30000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--skip-lr-scale', type=float, default=1.0, help='0 freezes the ridge skip')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--case-batches', action='store_true', help='official-style consecutive case records')
    parser.add_argument('--cond-weight', type=float, default=1.0)
    args = parser.parse_args()
    device = torch.device('cuda')
    tag = args.tag or args.name
    out = ROOT / 'runs/spatial' / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        print('exists', tag)
        return
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ridge = torch.load(ROOT / 'runs/ridge/ridge.pt', map_location='cpu', weights_only=True)
    model = make_spatial(args.name, ridge).to(device).float()
    fit, dev, val = Split('fit'), Split('dev'), Split('val')
    reader = Reader('fit', threads=args.workers)
    parameters = sum(p.numel() for p in model.parameters())
    print(tag, 'parameters', parameters, flush=True)
    groups = optimizer_groups(model, args.weight_decay)
    if args.skip_lr_scale != 1.0:
        skip_ids = {id(p) for n, p in model.named_parameters() if n.startswith('skip.') or 'linear_skip' in n}
        if args.skip_lr_scale == 0:
            for p in model.parameters():
                if id(p) in skip_ids:
                    p.requires_grad_(False)
            groups = [dict(g, params=[p for p in g['params'] if id(p) not in skip_ids]) for g in groups]
        else:
            split = []
            for g in groups:
                split.append(dict(g, params=[p for p in g['params'] if id(p) not in skip_ids]))
                split.append(dict(g, params=[p for p in g['params'] if id(p) in skip_ids], lr_scale=args.skip_lr_scale))
            groups = split
    groups = [g for g in groups if g['params']]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)

    if args.updates == 0:
        trainable = False
    else:
        trainable = True
    history = []
    started = time.monotonic()
    step = 0
    epoch = 0
    best_state = None
    best = math.inf
    cond_mask = fit.meta['cond'].numpy()
    while trainable and step < args.updates:
        rng = np.random.default_rng(args.seed + epoch)
        if args.case_batches:
            # Official-style: shuffled cases, records kept consecutive within a case.
            cases = fit.meta['case_id'].numpy()
            starts = np.flatnonzero(np.r_[True, cases[1:] != cases[:-1]])
            ends = np.r_[starts[1:], len(cases)]
            order = np.concatenate([np.arange(starts[i], ends[i]) for i in rng.permutation(len(starts))])
        else:
            order = rng.permutation(fit.n)
        for a, b in reader.batches(order, args.batch):
            x, y = decode(a, device), decode(b, device)
            lr = args.lr * min(1, (step + 1) / 1000) * (0.5 * (1 + math.cos(math.pi * min(step / args.updates, 1))) * (1 - 1e-5 / args.lr) + 1e-5 / args.lr)
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_scale', 1.0)
            loss = (model(x) - y).square().mean() / SCALES['spatial'] ** 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 5000 == 0 or step == args.updates:
                row = dict(step=step, train=float(loss.detach()), dev=score(model, dev, device, stride=4), val=score(model, val, device),
                           seconds=round(time.monotonic() - started, 1))
                history.append(row)
                print(json.dumps(row), flush=True)
                if row['val']['all'] < best:
                    best = row['val']['all']
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if step >= args.updates:
                break
        epoch += 1
    if best_state is not None:
        model.load_state_dict(best_state)
    final_dev, dev_errors = score(model, dev, device, keep=True)
    final_val, val_errors = score(model, val, device, keep=True)
    torch.save(dict(name=args.name, state=model.state_dict()), out / 'model.pt')
    torch.save(dict(dev=dev_errors.float(), val=val_errors.float()), out / 'errors.pt')
    (out / 'result.json').write_text(json.dumps(dict(args=vars(args), parameters=parameters, dev=final_dev, val=final_val,
                                                     history=history, seconds=time.monotonic() - started), indent=1))
    print(tag, 'final dev', final_dev, 'val', final_val, flush=True)


if __name__ == '__main__':
    main()
