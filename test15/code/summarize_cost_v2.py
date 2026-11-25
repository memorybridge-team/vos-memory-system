#!/usr/bin/env python3
"""Gates G1-G6 and the tables of DESIGN-cost-v2.md, from checks/v2/<method>.json.

  python3 code/summarize_cost_v2.py [--dir checks/v2] [--out cost_v2_summary]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ['resume', 'paper', 'conv_strip_vm', 'restart_p0', 'restart_mask', 'restart_p0_mask', 'olv', 'lastvis']
MEMORY = ['resume', 'paper', 'conv_strip_vm']
PROMPT = ['restart_p0', 'restart_mask', 'restart_p0_mask', 'olv', 'lastvis']
SETS = ['pumavos', 'm3vos']
DRAWS = 10_000


def key(r):
    return r['dataset'], r['video'], r['object'], r['fraction']


def boot(rows_by_video, stat, seed=0):
    """Video-clustered bootstrap 95% CI of stat(list of values)."""
    videos = list(rows_by_video)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(DRAWS):
        pick = rng.integers(0, len(videos), len(videos))
        out.append(stat([v for i in pick for v in rows_by_video[videos[i]]]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def by_video(rows, value):
    d = defaultdict(list)
    for r in rows:
        d[r['video']].append(value(r))
    return d


def fmt(x, ci=None, nd=2):
    s = f'{x:.{nd}f}'
    return s + (f' [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]' if ci else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='checks/v2')
    ap.add_argument('--out', default='cost_v2_summary')
    ap.add_argument('--reference', default='checks/single')
    args = ap.parse_args()
    d = Path(args.dir)
    data = {m: json.loads((d / f'{m}.json').read_text()) for m in METHODS if (d / f'{m}.json').exists()}
    rows = {m: {key(r): r for r in p['rows']} for m, p in data.items()}
    md, js = [], dict(gates={}, tables={})

    # ---- gates
    expected = set().union(*[set(v) for v in rows.values()]) if rows else set()
    g = js['gates']
    g['G1'] = {m: sum(1 for r in rows[m].values() if r['gpu_before']['other_pids'] or r['gpu_after']['other_pids']) for m in rows}
    g2 = {}
    for m in rows:
        checked = [r for r in rows[m].values() if r['pixels_s1_accuracy_run'] is not None]
        bad = [key(r) for r in checked if r['pixels_s1'] != r['pixels_s1_accuracy_run']]
        g2[m] = dict(checked=len(checked), mismatched=len(bad), examples=[list(b) for b in bad[:5]],
                     max_abs_pixel_diff=max([abs(r['pixels_s1'] - r['pixels_s1_accuracy_run']) for r in checked], default=0))
    g['G2'] = g2
    g['G3'] = {m: dict(complete=bool(data[m].get('complete')), rows=len(rows[m]), missing=len(expected - set(rows[m]))) for m in rows}
    g['G3']['methods_present'] = sorted(rows)
    g4 = {}
    for m, p in data.items():
        s0 = p['sentinels'][0]['ms']
        dev = [abs(s['ms'] / s0 - 1) for s in p['sentinels']]
        g4[m] = dict(start_ms=s0, max_rel_dev=max(dev), n=len(dev), ok=max(dev) <= 0.05)
    g['G4'] = g4
    g['G5'] = {}
    for m in rows:
        rel = [(np.percentile(r['times_ms'], 75) - np.percentile(r['times_ms'], 25)) / r['median_ms'] for r in rows[m].values()]
        g['G5'][m] = dict(frac_over_0_1=float(np.mean([x > 0.1 for x in rel])), median_rel_iqr=float(np.median(rel)))
    g6 = {}
    ref_dir = Path(args.reference)
    for m in rows:
        f = ref_dir / f'{m}.json'
        if not f.exists():
            continue
        ref = {(c['dataset'], c['video']): c['median_ms'] for c in json.loads(f.read_text())['cases']}
        pairs = [(r['median_ms'], ref[(r['dataset'], r['video'])]) for r in rows[m].values()
                 if r['fraction'] == '50' and r['object'] == 1 and (r['dataset'], r['video']) in ref]
        if pairs:
            g6[m] = dict(n=len(pairs), max_rel_diff=max(abs(a / b - 1) for a, b in pairs))
    g['G6'] = g6

    md.append('# 전환 비용 v2 요약\n')
    md.append(f'자동 생성(`code/summarize_cost_v2.py`). 입력 `{args.dir}`. 설계는 `DESIGN-cost-v2.md`.\n')
    md.append('## 검사\n')
    md.append('| 방법 | G1 위반 | G2 확인 / 불일치 (최대 픽셀 차) | G3 전환 / 빠짐 / 완료 | G4 최대 드리프트 | G5 IQR/중앙값 > 0.1 비율 | G6 최대 상대 차 |')
    md.append('|---|---:|---|---|---:|---:|---:|')
    for m in rows:
        g6s = f"{100 * g6[m]['max_rel_diff']:.1f}%" if m in g6 else '—'
        done = '예' if g['G3'][m]['complete'] else '아니오'
        md.append(f"| {m} | {g['G1'][m]} | {g2[m]['checked']} / {g2[m]['mismatched']} ({g2[m]['max_abs_pixel_diff']}) | "
                  f"{g['G3'][m]['rows']} / {g['G3'][m]['missing']} / {done} | "
                  f"{100 * g4[m]['max_rel_dev']:.1f}% | {100 * g['G5'][m]['frac_over_0_1']:.1f}% | {g6s} |")
    md.append('')

    # ---- table 1: time, memory, bytes
    t1 = {}
    md.append('## 1. 전환 1회 준비 시간 (ms, 전환별 20회 중앙값의 세트 중앙값 [영상 bootstrap 95%])\n')
    md.append('| 방법 | PUMaVOS | M3-VOS 전체 | M3-VOS 앞 81 영상 | 합친 값 | IQR (P / M) | p95 (P / M) | 최대 추가 메모리 MB | 넘기는 데이터 MB |')
    md.append('|---|---|---|---|---|---|---|---:|---:|')
    m3_81 = None
    acc_dir = Path('runs/unseen/m3vos/cases')
    if acc_dir.exists():
        m3_81 = {json.loads(p.read_text())['video'] for p in acc_dir.glob('*.json')}
    for m in rows:
        e = {}
        for s in SETS + (['m3vos81'] if m3_81 else []):
            rs = [r for r in rows[m].values() if r['dataset'] == (s if s != 'm3vos81' else 'm3vos') and (s != 'm3vos81' or r['video'] in m3_81)]
            if not rs:
                continue
            v = [r['median_ms'] for r in rs]
            e[s] = dict(n=len(rs), median=float(np.median(v)), ci=boot(by_video(rs, lambda r: r['median_ms']), np.median),
                        iqr=[float(np.percentile(v, 25)), float(np.percentile(v, 75))], p95=float(np.percentile(v, 95)),
                        peak_alloc_mb=float(max(r['peak_alloc_mb'] for r in rs)), bytes_mb=float(np.median([r['handoff_bytes'] for r in rs])) / 2 ** 20)
        if 'pumavos' in e and 'm3vos' in e:
            e['pooled'] = dict(median=(e['pumavos']['median'] + e['m3vos']['median']) / 2)
        t1[m] = e
        c = lambda s: fmt(e[s]['median'], e[s]['ci']) if s in e else '—'
        present = [s for s in SETS if s in e]
        iqr = ' / '.join(fmt(e[s]['iqr'][0]) + '–' + fmt(e[s]['iqr'][1]) for s in present)
        p95 = ' / '.join(fmt(e[s]['p95']) for s in present)
        peak = max(e[s]['peak_alloc_mb'] for s in present)
        size = ' / '.join(f"{e[s]['bytes_mb']:.2f}" for s in present)
        pooled = fmt(e['pooled']['median']) if 'pooled' in e else '—'
        md.append(f"| {m} | {c('pumavos')} | {c('m3vos')} | {c('m3vos81')} | {pooled} | {iqr} | {p95} | {peak:.0f} | {size} |")
    js['tables']['time'] = t1
    md.append('')

    # ---- table 2: stages
    md.append('## 2. 단계별 중앙값 (ms, 두 세트 합쳐서)\n')
    st = {}
    for m in rows:
        acc = defaultdict(list)
        for r in rows[m].values():
            for k, v in r['stages_ms'].items():
                acc[k].append(float(np.median(v)))
        st[m] = {k: float(np.median(v)) for k, v in acc.items()}
        md.append(f"- **{m}**: " + ', '.join(f'{k} {v:.2f}' for k, v in st[m].items()))
    js['tables']['stages'] = st
    md.append('')

    # ---- table 3: C1 ratios
    md.append('## 3. C1 비율: 재시작 시간 / 메모리 방법 시간 (전환별 비율의 중앙값 [영상 bootstrap 95%])\n')
    md.append('| 메모리 방법 | 재시작 | PUMaVOS | M3-VOS | 합친 값 |')
    md.append('|---|---|---|---|---|')
    c1 = {}
    for a in [x for x in MEMORY if x in rows]:
        for b in [x for x in PROMPT if x in rows]:
            e = {}
            for s in SETS:
                ks = [k for k in rows[a] if k[0] == s and k in rows[b]]
                if not ks:
                    continue
                rs = [dict(video=k[1], ratio=rows[b][k]['median_ms'] / rows[a][k]['median_ms']) for k in ks]
                e[s] = dict(n=len(ks), median=float(np.median([r['ratio'] for r in rs])), ci=boot(by_video(rs, lambda r: r['ratio']), np.median))
            if len(e) == 2:
                e['pooled'] = (e['pumavos']['median'] + e['m3vos']['median']) / 2
            c1[f'{b}/{a}'] = e
            md.append(f"| {a} | {b} | {fmt(e['pumavos']['median'], e['pumavos']['ci'], 1) if 'pumavos' in e else '—'} | "
                      f"{fmt(e['m3vos']['median'], e['m3vos']['ci'], 1) if 'm3vos' in e else '—'} | {fmt(e['pooled'], nd=1) if 'pooled' in e else '—'} |")
    js['tables']['C1'] = c1
    md.append('')

    # ---- table 4: C2 frame equivalents
    md.append('## 4. C2: 한 프레임 처리 시간과 프레임 환산\n')
    frame = {}
    for m, p in data.items():
        tr = 'small' if m in ('paper', 'conv_strip_vm', 'olv', 'lastvis') else 'base'
        frame.setdefault(tr, []).extend(t['median_ms'] for t in p.get('tracker_steps', []))
    frame = {k: float(np.median(v)) for k, v in frame.items() if v}
    js['tables']['frame_ms'] = frame
    md.append('추적 한 단계(case별 중앙값의 중앙값): ' + ', '.join(f'{k} {v:.2f} ms' for k, v in frame.items()) + '\n')
    if 'base' in frame:
        md.append('| 방법 | 합친 값 (ms) | Base+ 프레임 환산 |')
        md.append('|---|---:|---:|')
        for m in rows:
            if 'pooled' in t1[m]:
                md.append(f"| {m} | {t1[m]['pooled']['median']:.2f} | {t1[m]['pooled']['median'] / frame['base']:.2f} |")
    md.append('')

    # ---- table 5: cold
    md.append('## 5. 첫 호출 (ms)\n')
    md.append('| 방법 | 프로세스 첫 전환 cold | 나머지 전환 cold 중앙값 / 최대 |')
    md.append('|---|---:|---|')
    cold = {}
    for m in rows:
        first = [r['cold_ms'] for r in rows[m].values() if r['first_in_process']]
        rest = [r['cold_ms'] for r in rows[m].values() if not r['first_in_process']]
        cold[m] = dict(first=first, rest_median=float(np.median(rest)) if rest else None, rest_max=float(max(rest)) if rest else None)
        md.append(f"| {m} | {', '.join(f'{x:.1f}' for x in first)} | {fmt(cold[m]['rest_median']) if rest else '—'} / {fmt(cold[m]['rest_max']) if rest else '—'} |")
    js['tables']['cold'] = cold
    md.append('')

    # ---- table 6: by fraction
    md.append('## 6. PUMaVOS 전환 지점별 중앙값 (ms)\n')
    md.append('| 방법 | 25% | 50% | 75% |')
    md.append('|---|---:|---:|---:|')
    for m in rows:
        v = {f: [r['median_ms'] for r in rows[m].values() if r['dataset'] == 'pumavos' and r['fraction'] == f] for f in ('25', '50', '75')}
        md.append(f"| {m} | " + ' | '.join(fmt(float(np.median(v[f]))) if v[f] else '—' for f in ('25', '50', '75')) + ' |')
    md.append('')

    Path(args.out + '.md').write_text('\n'.join(md) + '\n')
    Path(args.out + '.json').write_text(json.dumps(js, indent=1))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
