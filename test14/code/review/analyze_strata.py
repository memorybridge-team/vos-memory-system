#!/usr/bin/env python3
"""Where the state error lives: presence, conditioning, age, dataset, shrinkage, tails."""
import json

import numpy as np
import torch

from review_data import ROOT, SCALES, Moments, Split

device = torch.device('cuda')
OUT = ROOT / 'runs/analysis'
report = {}


def show(title, value):
    print(f'--- {title}\n{json.dumps(value, indent=None, default=float)}', flush=True)
    report[title] = value


def stratum(rows, mask):
    if not bool(mask.any()):
        return None
    ss, sp = SCALES['spatial'] ** 2, SCALES['pointer'] ** 2
    return dict(n=int(mask.sum()), share=float(mask.float().mean()),
                sp_ridge=float(rows['sp_ridge'][mask].mean() / ss), sp_legacy=float(rows['sp_legacy'][mask].mean() / ss),
                p_ridge=float(rows['p_ridge'][mask].mean() / sp), p_legacy=float(rows['p_legacy'][mask].mean() / sp),
                sp_direct=float(rows['sp_direct'][mask].mean() / ss), p_direct=float(rows['p_direct'][mask].mean() / sp))


def main():
    splits = {k: Split(k) for k in ('fit', 'dev', 'val')}
    ridge = torch.load(ROOT / 'runs/ridge/ridge.pt', map_location=device, weights_only=True)
    # Presence agreement and the no-object pointer.
    for name, split in splits.items():
        m = split.meta
        sp, tp = m['src_pres'] > 0, m['tgt_pres'] > 0
        lvos = split.is_lvos
        for label, mask in (('MOSE', ~lvos), ('LVOS', lvos)):
            if not bool(mask.any()):
                continue
            n = int(mask.sum())
            show(f'{name} {label} presence (src,tgt)', dict(
                n=n, both=float((sp & tp & mask).sum() / n), src_only=float((sp & ~tp & mask).sum() / n),
                tgt_only=float((~sp & tp & mask).sum() / n), neither=float((~sp & ~tp & mask).sum() / n)))
    x, y = splits['fit'].pointers(device)
    m = splits['fit'].meta
    sa, ta = (m['src_pres'] <= 0).to(device), (m['tgt_pres'] <= 0).to(device)
    cond = m['cond'].to(device)
    show('fit absent pointers', dict(
        src_absent_n=int(sa.sum()), src_absent_std_max=float(x[sa].std(0).max()), src_absent_std_mean=float(x[sa].std(0).mean()),
        tgt_absent_n=int(ta.sum()), tgt_absent_std_max=float(y[ta].std(0).max()), tgt_absent_std_mean=float(y[ta].std(0).mean()),
        present_std_mean=float(x[~sa].std(0).mean()), tgt_present_std_mean=float(y[~ta].std(0).mean()),
        src_const_norm=float(x[sa].mean(0).norm()), tgt_const_norm=float(y[ta].mean(0).norm()),
        present_mean_norm=float(x[~sa].mean(0).norm()), tgt_present_mean_norm=float(y[~ta].mean(0).norm())))
    c_s, c_t = x[sa].mean(0), y[ta].mean(0)
    torch.save(dict(c_s=c_s.cpu(), c_t=c_t.cpu()), OUT / 'no_object_pointers.pt')
    # Conditioning pointer structure.
    show('fit pointer cond', dict(cond_n=int(cond.sum()), cond_std=float(y[cond].std(0).mean()), rest_std=float(y[~cond].std(0).mean()),
                                  cond_mean_dist=float((y[cond].mean(0) - y[~cond].mean(0)).norm())))
    # Per-record strata for legacy and ridge.
    for name in ('val', 'dev'):
        split = splits[name]
        rows = torch.load(OUT / f'{name}_records.pt', map_location='cpu', weights_only=True)
        meta = split.meta
        cond = meta['cond']
        age = meta['switch'] - meta['frame']
        sp, tp = meta['src_pres'] > 0, meta['tgt_pres'] > 0
        lvos = split.is_lvos
        all_mask = torch.ones(split.n, dtype=torch.bool)
        show(f'{name} all', stratum(rows, all_mask))
        for label, mask in (('LVOS', lvos), ('MOSE', ~lvos)):
            if bool(mask.any()):
                show(f'{name} {label}', stratum(rows, mask))
                show(f'{name} {label} cond', stratum(rows, mask & cond))
                show(f'{name} {label} recent 0-5', stratum(rows, mask & ~cond & (age <= 5)))
                show(f'{name} {label} older 6+', stratum(rows, mask & ~cond & (age > 5)))
                show(f'{name} {label} both present', stratum(rows, mask & sp & tp))
                show(f'{name} {label} neither present', stratum(rows, mask & ~sp & ~tp))
                show(f'{name} {label} src only', stratum(rows, mask & sp & ~tp))
                show(f'{name} {label} tgt only', stratum(rows, mask & ~sp & tp))
        # Shrinkage and cosine.
        show(f'{name} energy', dict(
            sp_ridge_ratio=float(rows['sp_ridge_energy'].mean() / rows['sp_tgt_energy'].mean()),
            sp_legacy_ratio=float(rows['sp_legacy_energy'].mean() / rows['sp_tgt_energy'].mean()),
            p_ridge_ratio=float(rows['p_ridge_energy'].mean() / rows['p_tgt_energy'].mean()),
            p_legacy_ratio=float(rows['p_legacy_energy'].mean() / rows['p_tgt_energy'].mean()),
            sp_ridge_cos=float(rows['sp_ridge_cos'].mean()), sp_legacy_cos=float(rows['sp_legacy_cos'].mean()),
            p_ridge_cos=float(rows['p_ridge_cos'].mean()), p_legacy_cos=float(rows['p_legacy_cos'].mean())))
        # Error by time since the prompt: does independent rollout divergence grow?
        prompt = meta['frame'][meta['cond']]  # one conditioning record per case
        case = meta['case_id'].long()
        first = torch.full((len(meta['cases']),), -1, dtype=torch.long)
        first[case[cond]] = meta['frame'][cond].long()
        since = meta['frame'].long() - first[case]
        buckets = [(1, 15), (16, 60), (61, 200), (201, 100000)]
        show(f'{name} error by frames since prompt', {f'{a}-{b}': stratum(rows, ~cond & (since >= a) & (since <= b) & sp & tp) for a, b in buckets})
        # Heavy tail: share of the total SSE held by the worst cases.
        cases = len(meta['cases'])
        for key in ('sp_ridge', 'p_ridge', 'p_legacy', 'sp_legacy'):
            total = torch.zeros(cases, dtype=torch.float64).index_add_(0, case, rows[key])
            count = torch.zeros(cases, dtype=torch.float64).index_add_(0, case, torch.ones_like(rows[key]))
            value = (total / count).sort(descending=True).values
            show(f'{name} tail {key}', dict(top5pct_share=float(value[:max(1, cases // 20)].sum() / value.sum()),
                                            median=float(value.median()), p90=float(value[cases // 10]), maxv=float(value[0])))
    # Variance split of the target pointer: identity of the object vs. change within a case.
    for name in ('fit', 'val'):
        split = splits[name]
        cases = split.meta['case_id'].long().to(device)
        keep = (split.meta['src_pres'] > 0).to(device) & (split.meta['tgt_pres'] > 0).to(device) & ~split.meta['cond'].to(device)
        x, y = split.pointers(device)
        x, y, c = x[keep].double(), y[keep].double(), cases[keep]
        count = torch.zeros(int(c.max()) + 1, dtype=torch.float64, device=device).index_add_(0, c, torch.ones(len(c), dtype=torch.float64, device=device))
        def within(v):
            s = torch.zeros(int(c.max()) + 1, 256, dtype=torch.float64, device=device).index_add_(0, c, v)
            return v - (s / count.clamp_min(1)[:, None])[c]
        ym, xm = y.mean(0), x.mean(0)
        pred = x @ ridge['pointer_w'] + ridge['pointer_b']
        ew = within(pred - y)
        total, inside = (y - ym).square().sum(), within(y).square().sum()
        err_total, err_inside = (pred - y).square().sum(), ew.square().sum()
        show(f'{name} present pointer variance', dict(between_share=float(1 - inside / total), within_share=float(inside / total),
             ridge_r2_total=float(1 - err_total / total), ridge_r2_between=float(1 - (err_total - err_inside) / (total - inside)),
             ridge_r2_within=float(1 - err_inside / inside)))
    (OUT / 'strata.json').write_text(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main()
