#!/usr/bin/env python3
"""Baselines and data structure: analysis ridge (fit only), legacy Transformer, per-record errors."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from review_data import ROOT, SCALES, Moments, Split, chunks
from vos_memory_inspector.state_training.trainer import restore_model

ALPHAS = (0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1)
LEGACY = Path('/workspace/j700qtcgrg/CMMT-official-isolated/official-20261005T012256KST/'
              'final-state-loss-20261005T1627KST/delivery/selected_state_loss_best/translator_weights.pth')
OUT = ROOT / 'runs/analysis'
device = torch.device('cuda')


def spatial_moments(split):
    m = Moments(64, 64, device)
    started = time.time()
    for i, index in enumerate(chunks(split.n, 256)):
        x, y = split.spatial(index, device)
        m.update(x.movedim(1, -1).reshape(-1, 64), y.movedim(1, -1).reshape(-1, 64))
        if i % 200 == 0:
            print(split.name, 'spatial', index[0], f'{time.time() - started:.0f}s', flush=True)
    return m


def pointer_moments(split, mask=None):
    x, y = split.pointers(device)
    if mask is not None:
        x, y = x[mask.to(device)], y[mask.to(device)]
    m = Moments(256, 256, device)
    m.update(x, y)
    return m


def fit_ridge(fit, dev, name):
    best = None
    rows = []
    for alpha in ALPHAS:
        w, b = fit.solve(alpha)
        score = dev.sse(w, b) / dev.n / (64 if name == 'spatial' else 256)
        rows.append(dict(alpha=alpha, dev_mse=score))
        if best is None or score < best[0] * (1 - 1e-8):
            best = (score, alpha, w, b)
    print(name, rows, 'selected', best[1], flush=True)
    return best[1], best[2], best[3], rows


def per_record(split, ridge, legacy):
    """Per-record SSE for direct/ridge/legacy plus energy, stored for strata analysis."""
    n = split.n
    rows = {k: torch.zeros(n, dtype=torch.float64) for k in (
        'sp_direct', 'sp_ridge', 'sp_legacy', 'p_direct', 'p_ridge', 'p_legacy',
        'sp_tgt_energy', 'sp_ridge_energy', 'sp_legacy_energy', 'p_tgt_energy', 'p_ridge_energy', 'p_legacy_energy',
        'sp_ridge_cos', 'sp_legacy_cos', 'p_ridge_cos', 'p_legacy_cos')}
    psrc, ptgt = split.pointers(device)
    with torch.inference_mode():
        for index in chunks(n, 128):
            i = torch.from_numpy(index)
            x, y = split.spatial(index, device)
            p, q = psrc[i.to(device)], ptgt[i.to(device)]
            r = (x.movedim(1, -1).double() @ ridge['spatial_w'] + ridge['spatial_b']).movedim(-1, 1).float()
            s = (p.double() @ ridge['pointer_w'] + ridge['pointer_b']).float()
            a, b = legacy.translate_tensors(x[None, None], p[None, None])
            a, b = a[0, 0], b[0, 0]
            rows['sp_direct'][i] = (x - y).square().flatten(1).mean(1).double().cpu()
            rows['sp_ridge'][i] = (r - y).square().flatten(1).mean(1).double().cpu()
            rows['sp_legacy'][i] = (a - y).square().flatten(1).mean(1).double().cpu()
            rows['p_direct'][i] = (p - q).square().mean(1).double().cpu()
            rows['p_ridge'][i] = (s - q).square().mean(1).double().cpu()
            rows['p_legacy'][i] = (b - q).square().mean(1).double().cpu()
            rows['sp_tgt_energy'][i] = y.square().flatten(1).mean(1).double().cpu()
            rows['sp_ridge_energy'][i] = r.square().flatten(1).mean(1).double().cpu()
            rows['sp_legacy_energy'][i] = a.square().flatten(1).mean(1).double().cpu()
            rows['p_tgt_energy'][i] = q.square().mean(1).double().cpu()
            rows['p_ridge_energy'][i] = s.square().mean(1).double().cpu()
            rows['p_legacy_energy'][i] = b.square().mean(1).double().cpu()
            cos = torch.nn.functional.cosine_similarity
            rows['sp_ridge_cos'][i] = cos(r, y, dim=1).flatten(1).mean(1).double().cpu()
            rows['sp_legacy_cos'][i] = cos(a, y, dim=1).flatten(1).mean(1).double().cpu()
            rows['p_ridge_cos'][i] = cos(s, q, dim=1).double().cpu()
            rows['p_legacy_cos'][i] = cos(b, q, dim=1).double().cpu()
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'runs/ridge').mkdir(parents=True, exist_ok=True)
    splits = {k: Split(k) for k in ('fit', 'dev', 'val')}
    print({k: v.n for k, v in splits.items()}, flush=True)
    moments_path = OUT / 'spatial_moments.pt'
    if moments_path.exists():
        saved = torch.load(moments_path, map_location=device, weights_only=False)
    else:
        saved = {}
        for name in ('fit', 'dev', 'val'):
            m = spatial_moments(splits[name])
            saved[name] = {k: getattr(m, k) for k in ('n', 'sx', 'sy', 'xx', 'xy', 'yy')}
        torch.save(saved, moments_path)
    sm = {}
    for name, values in saved.items():
        m = Moments(64, 64, device)
        for k, v in values.items():
            setattr(m, k, v)
        sm[name] = m
    pm = {k: pointer_moments(v) for k, v in splits.items()}
    alpha_s, ws, bs, rows_s = fit_ridge(sm['fit'], sm['dev'], 'spatial')
    alpha_p, wp, bp, rows_p = fit_ridge(pm['fit'], pm['dev'], 'pointer')
    ptgt = splits['fit'].pointers()[1]
    ridge = dict(spatial_w=ws, spatial_b=bs, pointer_w=wp, pointer_b=bp, alpha_spatial=alpha_s, alpha_pointer=alpha_p,
                 target_pointer_mean=ptgt.mean(0).to(device).double(), candidates=dict(spatial=rows_s, pointer=rows_p))
    torch.save({k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in ridge.items()}, ROOT / 'runs/ridge/ridge.pt')
    summary = dict(alpha=dict(spatial=alpha_s, pointer=alpha_p), splits={})
    for name in ('fit', 'dev', 'val'):
        s, p = sm[name], pm[name]
        zero = lambda d: (torch.zeros(d, d, dtype=torch.float64, device=device), torch.zeros(d, dtype=torch.float64, device=device))
        summary['splits'][name] = dict(
            records=splits[name].n,
            spatial_nmse_ridge=s.sse(ws, bs) / s.n / 64 / SCALES['spatial'] ** 2,
            spatial_nmse_mean=s.sse(*zero(64)[:1], s.sy / s.n) / s.n / 64 / SCALES['spatial'] ** 2,
            pointer_nmse_ridge=p.sse(wp, bp) / p.n / 256 / SCALES['pointer'] ** 2,
            pointer_nmse_mean=p.sse(zero(256)[0], p.sy / p.n) / p.n / 256 / SCALES['pointer'] ** 2)
    print(json.dumps(summary, indent=1), flush=True)
    (OUT / 'summary_ridge.json').write_text(json.dumps(summary, indent=1))

    raw = torch.load(LEGACY, map_location='cpu', weights_only=True)
    legacy = restore_model('transformer', raw.get('transformer', raw)).to(device).float().eval()
    for name in ('val', 'dev'):
        rows = per_record(splits[name], ridge, legacy)
        torch.save(rows, OUT / f'{name}_records.pt')
        print(name, 'per-record saved', flush=True)
    # Spectra of source/target second moments.
    spectra = {}
    for name, m in (('spatial', sm['fit']), ('pointer', pm['fit'])):
        d = m.xx.shape[0]
        mx, my = m.sx / m.n, m.sy / m.n
        g = m.xx / m.n - torch.outer(mx, mx)
        e = torch.linalg.eigvalsh(g).flip(0).clamp_min(0)
        spectra[name] = dict(source_eig=e.cpu().tolist(), source_var=float(e.sum()))
    (OUT / 'spectra.json').write_text(json.dumps(spectra))


if __name__ == '__main__':
    main()
