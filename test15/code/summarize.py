#!/usr/bin/env python3
"""Summary of memory-resume-20261120 (plan sections 3 and 4).

  python3 code/summarize.py --tag pumavos --tag m3vos --out summary      (writes summary.json and summary.md)

Scores are case-equal mean J&F x100 over GT-visible frames after the switch, as in every earlier table. CIs are 95%
video-clustered paired bootstrap, 10,000 draws, seed 0. The pooled estimate averages the datasets with equal weight
and resamples videos within each dataset. A win or loss is a case difference above 2 points.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

E = Path('/workspace/memory-resume-20261120/runs/unseen')
METHODS = ['base', 'resume', 'restart_p0', 'restart_mask', 'restart_p0_mask', 'olv', 'lastvis', 'paper', 'conv_strip_vm', 'small']
LABELS = dict(base='Base+ 단독', resume='이전 상태로 재개', restart_p0='재시작: 처음 프롬프트', restart_mask='재시작: 자기 마지막 마스크',
              restart_p0_mask='재시작: 처음 프롬프트 + 자기 마스크', olv='OLV (처음 프롬프트 + Small 마스크)', lastvis='Small 마스크만',
              paper='논문 translator (r2)', conv_strip_vm='conv_strip_vm', small='Small 단독')
CONTRASTS = [('resume', 'restart_p0_mask'), ('resume', 'restart_p0'), ('resume', 'restart_mask'),
             ('restart_p0_mask', 'restart_p0'), ('restart_p0_mask', 'restart_mask'), ('olv', 'restart_p0_mask'),
             ('lastvis', 'restart_mask'), ('paper', 'restart_p0'), ('paper', 'restart_p0_mask'), ('paper', 'resume'),
             ('conv_strip_vm', 'restart_p0'), ('conv_strip_vm', 'restart_p0_mask'), ('conv_strip_vm', 'resume')]
RECOVERY = ['paper', 'conv_strip_vm', 'olv', 'restart_mask', 'restart_p0_mask']
WINDOWS = [(1, 6), (7, 16), (17, 100), (101, None)]
DRAWS = 10000
MARGIN = 1.5


def load(tag):
    out = []
    for f in sorted(glob.glob(str(E / tag / 'cases' / '*.json'))):
        d = json.loads(Path(f).read_text())
        for label, row in d['switches'].items():
            out.append(dict(dataset=d['dataset'], video=d['video'], object=d['object'], label=label, start=d['start'], row=row))
    return out


def window(tr, lo, hi):
    J, F, V = tr['J'], tr['F'], tr['visible']
    end = len(J) if hi is None else min(hi, len(J))
    vals = [(J[i] + F[i]) / 2 for i in range(lo - 1, end) if V[i]]
    return 100 * sum(vals) / len(vals) if vals else None


def false_positive(tr):
    empty = [a for a, v in zip(tr['area'], tr['visible']) if v is False]
    return (100 * sum(a > 0 for a in empty) / len(empty), len(empty)) if empty else (None, 0)


def draws(cases, columns, seed):
    """[DRAWS, len(columns)] case-equal means over videos resampled with replacement."""
    videos = sorted({c['video'] for c in cases})
    index = {v: i for i, v in enumerate(videos)}
    sums, counts = np.zeros((len(videos), len(columns))), np.zeros(len(videos))
    for c in cases:
        sums[index[c['video']]] += [c[k] for k in columns]
        counts[index[c['video']]] += 1
    picks = np.random.default_rng(seed).integers(0, len(videos), size=(DRAWS, len(videos)))
    return sums[picks].sum(1) / counts[picks].sum(1)[:, None]


def ci(x):
    return [float(np.quantile(x, .025)), float(np.quantile(x, .975))]


def values(rows, key_fn):
    out = []
    for r in rows:
        v = key_fn(r)
        if v is not None:
            out.append(v)
    return out


def contrast(rows, a, b, score, seed=0):
    cases = []
    for r in rows:
        x, y = score(r, a), score(r, b)
        if x is not None and y is not None:
            cases.append(dict(video=r['video'], d=x - y, x=x, y=y))
    if not cases:
        return None
    dist = draws(cases, ['d'], seed)[:, 0]
    d = np.array([c['d'] for c in cases])
    return dict(n=len(cases), videos=len({c['video'] for c in cases}), mean=float(d.mean()), ci=ci(dist),
                wins=int((d > 2).sum()), losses=int((d < -2).sum()),
                collapses=int(sum(c['x'] < 20 <= 50 <= c['y'] for c in cases)),
                rescues=int(sum(c['y'] < 20 <= 50 <= c['x'] for c in cases)), _dist=dist)


def full(r, m):
    v = r['row'].get(m)
    return 100 * v['JF'] if isinstance(v, dict) and v.get('JF') is not None else None


def verdict(lo, hi, per_dataset):
    if lo > 0:
        text = '메모리가 가장 강한 재시작보다 유리하다' + (f' (단, 차이는 {MARGIN} 이내)' if hi <= MARGIN else '')
        return text + ('; 한 데이터셋은 CI < 0' if any(c[1] < 0 for c in per_dataset) else '')
    if hi < 0:
        text = '재시작이 유리하다' + (f' (단, 차이는 {MARGIN} 이내)' if lo >= -MARGIN else '')
        return text + ('; 한 데이터셋은 CI > 0' if any(c[0] > 0 for c in per_dataset) else '')
    if lo >= -MARGIN and hi <= MARGIN:
        return f'같은 수준이다 (CI가 ±{MARGIN} 안)'
    return '결론 없음'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', action='append', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    data = {}
    for tag in args.tag:
        rows = load(tag)
        data[rows[0]['dataset'] if rows else tag] = rows
    report, md = dict(tags=args.tag, datasets={}), ['# memory-resume-20261120 요약', '',
                                                     f'점수는 case 평균 J&F ×100(전환 뒤 정답이 보이는 프레임). CI는 영상 단위 paired bootstrap 95%, {DRAWS:,}회. 승/패는 case 차이 2점 초과.', '']
    pooled = {}
    for ds, rows in data.items():
        out = report['datasets'][ds] = dict(cases=len(rows), videos=len({r['video'] for r in rows}))
        md += [f'## {ds} ({len(rows)} case, {out["videos"]} 영상)', '', '| 방법 | J&F | 이전 상태로 재개 대비 [95% CI] | 승/패 | 붕괴 |', '|---|---:|---:|---:|---:|']
        means = {m: values(rows, lambda r: full(r, m)) for m in METHODS}
        out['mean'] = {m: float(np.mean(v)) if v else None for m, v in means.items()}
        for m in METHODS:
            if not means[m]:
                continue
            c = contrast(rows, m, 'resume', full) if m != 'resume' else None
            cell = f"{c['mean']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]" if c else ''
            wl = f"{c['wins']}/{c['losses']}" if c else ''
            col = f"{c['collapses']}" if c else ''
            md.append(f"| {LABELS[m]} | {np.mean(means[m]):.2f} | {cell} | {wl} | {col} |")
        md += ['', '| 대비 | 차이 [95% CI] | case | 승/패 | 붕괴 / 회복 |', '|---|---:|---:|---:|---:|']
        out['contrasts'] = {}
        for a, b in CONTRASTS:
            c = contrast(rows, a, b, full)
            if c is None:
                continue
            pooled.setdefault((a, b), []).append(c['_dist'])
            out['contrasts'][f'{a}-{b}'] = {k: v for k, v in c.items() if k != '_dist'}
            md.append(f"| {a} − {b} | {c['mean']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}] | {c['n']} | {c['wins']}/{c['losses']} | {c['collapses']} / {c['rescues']} |")
        md += ['', '회복률 = (방법 − 처음 프롬프트 재시작) / (이전 상태로 재개 − 처음 프롬프트 재시작)', '', '| 방법 | 회복률 [95% CI] |', '|---|---:|']
        out['recovery'] = {}
        for m in RECOVERY:
            cases = [dict(video=r['video'], num=full(r, m) - full(r, 'restart_p0'), den=full(r, 'resume') - full(r, 'restart_p0'))
                     for r in rows if None not in (full(r, m), full(r, 'restart_p0'), full(r, 'resume'))]
            if not cases:
                continue
            dist = draws(cases, ['num', 'den'], 0)
            ratio = dist[:, 0] / dist[:, 1]
            point = np.mean([c['num'] for c in cases]) / np.mean([c['den'] for c in cases])
            out['recovery'][m] = dict(point=float(point), ci=ci(ratio))
            md.append(f"| {LABELS[m]} | {100 * point:.0f}% [{100 * ci(ratio)[0]:.0f}%, {100 * ci(ratio)[1]:.0f}%] |")
        md += ['', '시간 창별 J&F (전환 뒤 프레임 오프셋)', '', '| 방법 | ' + ' | '.join(f'+{lo}…{hi or ""}' for lo, hi in WINDOWS) + ' |',
               '|---|' + '---:|' * len(WINDOWS)]
        out['windows'] = {}
        for m in METHODS:
            cells = []
            for lo, hi in WINDOWS:
                score = (lambda r, mm, lo=lo, hi=hi: window(r['row']['trace'][mm], lo, hi) if mm in r['row']['trace'] else None)
                if m == 'resume':
                    v = values(rows, lambda r: score(r, m))
                    cells.append(f'{np.mean(v):.2f}' if v else '')
                    out['windows'].setdefault(m, {})[f'{lo}-{hi}'] = dict(mean=float(np.mean(v)) if v else None)
                    continue
                c = contrast(rows, m, 'resume', score)
                cells.append(f"{c['mean']:+.2f} [{c['ci'][0]:+.1f}, {c['ci'][1]:+.1f}]" if c else '')
                out['windows'].setdefault(m, {})[f'{lo}-{hi}'] = {k: v for k, v in c.items() if k != '_dist'} if c else None
            md.append(f'| {LABELS[m]}{" (평균)" if m == "resume" else " (재개 대비)"} | ' + ' | '.join(cells) + ' |')
        md += ['', 's에서 Base+ 최근 6칸 중 객체가 있는 칸 수로 나눈 층', '', '| 층 | case | 재개 − 처음 프롬프트 재시작 | 재개 − 처음 프롬프트 + 자기 마스크 |', '|---|---:|---:|---:|']
        out['strata'] = {}
        for name, fn in [('6칸', lambda k: k == 6), ('1–5칸', lambda k: 0 < k < 6), ('0칸', lambda k: k == 0)]:
            sub = [r for r in rows if fn(r['row']['base_recent6']['present'])]
            cells = []
            for b in ('restart_p0', 'restart_p0_mask'):
                c = contrast(sub, 'resume', b, full) if len({r['video'] for r in sub}) > 1 else None
                cells.append(f"{c['mean']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]" if c else ('n<2 영상' if sub else ''))
                out['strata'].setdefault(name, {})[b] = {k: v for k, v in c.items() if k != '_dist'} if c else None
            md.append(f'| {name} | {len(sub)} | ' + ' | '.join(cells) + ' |')
        md += ['', '전환 뒤 정답이 비어 있는 프레임의 오탐률 (예측 면적 > 0인 프레임 비율, case 평균)', '', '| 방법 | 오탐률 | case |', '|---|---:|---:|']
        out['false_positive'] = {}
        for m in METHODS:
            v = [false_positive(r['row']['trace'][m])[0] for r in rows if m in r['row']['trace']]
            v = [x for x in v if x is not None]
            out['false_positive'][m] = dict(mean=float(np.mean(v)) if v else None, cases=len(v))
            if v:
                md.append(f'| {LABELS[m]} | {np.mean(v):.1f}% | {len(v)} |')
        curve = {}
        for m in METHODS:
            if m == 'resume':
                continue
            diffs = []
            for k in range(1, 101):
                d = [window(r['row']['trace'][m], k, k) - window(r['row']['trace']['resume'], k, k) for r in rows
                     if m in r['row']['trace'] and window(r['row']['trace']['resume'], k, k) is not None
                     and window(r['row']['trace'][m], k, k) is not None]
                diffs.append(float(np.mean(d)) if d else None)
            curve[m] = diffs
        out['curve_vs_resume_offsets_1_100'] = curve
        md.append('')
    if len(data) > 1:
        md += ['## 데이터셋을 같은 가중치로 합친 값', '', '| 대비 | 차이 [95% CI] | 데이터셋 |', '|---|---:|---:|']
        report['pooled'] = {}
        for (a, b), dists in pooled.items():
            if len(dists) != len(data):
                continue
            mean_each = [report['datasets'][ds]['contrasts'][f'{a}-{b}']['mean'] for ds in data]
            dist = np.mean(np.stack(dists), 0)
            report['pooled'][f'{a}-{b}'] = dict(mean=float(np.mean(mean_each)), ci=ci(dist))
            md.append(f"| {a} − {b} | {np.mean(mean_each):+.2f} [{ci(dist)[0]:+.2f}, {ci(dist)[1]:+.2f}] | {len(dists)} |")
        main_ci = report['pooled'].get('resume-restart_p0_mask', {}).get('ci')
        if main_ci:
            each = [report['datasets'][ds]['contrasts']['resume-restart_p0_mask']['ci'] for ds in data]
            report['verdict'] = verdict(main_ci[0], main_ci[1], each)
            md += ['', f"**판정 (주 대비 resume − restart_p0_mask, 합친 값):** {report['verdict']}", '']
    Path(args.out + '.json').write_text(json.dumps(report, indent=1))
    Path(args.out + '.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
