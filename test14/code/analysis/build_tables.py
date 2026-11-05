#!/usr/bin/env python3
"""Builds every table of test14 from the raw results. No model runs.

Usage: build_tables.py --runs <folder with jf/, unseen/, spatial/, ...> --paper <runpod_import folder> --out <results/tables>

Scores are x100. A case score averages visible frames after the switch; cases then average within a video, and videos average.
Differences are paired on the same cases. Intervals bootstrap over videos, 20,000 draws, a fresh seed 7 for each comparison.
"""
import argparse
import csv
import glob
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

# Where a method was run more than once on one group, the first tag in this order wins. t1 and t4 are last because they ran
# under the older bf16 continuation or a separate baseline round; the order is the one the review reports used.
PRIORITY = ('t2', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't4', 't1')
GROUPS = {'mose': 'MOSE dev', 'val': 'LVOS val', 'lvd': 'LVOS dev'}
WINDOWS = {'jf_1_10': (1, 10), 'jf_11_30': (11, 30), 'jf_31_100': (31, 100), 'jf_101_end': (101, 10 ** 9)}
REFERENCES = ('target_ref', 'legacy', 'mlp_context', 'e2e_ctx_f')
PAPER = {'source_only': 'Source-only', 'full_replay': 'Full Replay', 'original_last_visible': 'Original+Last-Visible', 'direct_state_copy': 'Direct Copy',
         'original_replay_8': 'Replay-8', 'translator': 'paper Translator'}
REVIEW_FOR_PAPER = ('small', 'base', 'olv', 'direct', 'legacy', 'mlp_context', 'e2e_mlp_f', 'e2e_tf_f', 'e2e_ctx_f', 'e2e3_ctx_f', 'e2e_ctx_pin7', 'e2e3_ctx_pin5')


def write(path, header, rows):
    with open(path, 'w', newline='') as handle:
        out = csv.writer(handle, lineterminator='\n')
        out.writerow(header)
        out.writerows(rows)
    print(f'{path.name}: {len(rows)} rows')


def num(value, digits=4):
    return '' if value is None else round(float(value), digits)


def bootstrap(diff, seed=7, draws=20000):
    d = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    s = d[rng.integers(0, len(d), (draws, len(d)))].mean(1)
    low, high = np.percentile(s, [2.5, 97.5])
    return float(d.mean()), float(low), float(high)


def video_means(pairs):
    grouped = defaultdict(list)
    for video, value in pairs:
        if value is not None:
            grouped[video].append(value)
    return {k: float(np.mean(v)) for k, v in grouped.items()}


def paired(a, b):
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return None
    d = np.array([a[k] - b[k] for k in keys])
    mean, low, high = bootstrap(d)
    return dict(diff=mean, low=low, high=high, videos=len(keys), better=int((d > 1).sum()), worse=int((d < -1).sum()))


def frame_scores(row, method):
    s = row['scores'][method]
    return [(o, j, f) for o, j, f, v in zip(row['offsets'], s['J'], s['F'], s['visible']) if o >= 1 and j is not None and v]


def case_row(row, method):
    frames = frame_scores(row, method)
    if not frames:
        return None
    out = dict(n=len(frames), j=100 * float(np.mean([x[1] for x in frames])), f=100 * float(np.mean([x[2] for x in frames])))
    out['jf'] = (out['j'] + out['f']) / 2
    for name, (low, high) in WINDOWS.items():
        got = [(j + f) / 2 for o, j, f in frames if low <= o <= high]
        out[name] = 100 * float(np.mean(got)) if got else None
    return out


def group_of(tag):
    name, _, key = tag.rpartition('_')
    if name == 't20':
        return 'MOSE fit' if key == 'mose' else 'LVOS fit'
    return GROUPS.get(key)


def read_jf(runs):
    rows = []
    for folder in sorted(glob.glob(f'{runs}/jf/*')):
        tag = os.path.basename(folder)
        for path in sorted(glob.glob(f'{folder}/cases/*.json')):
            row = json.load(open(path))
            case = row['case']
            for method in row['scores']:
                got = case_row(row, method)
                if got is not None:
                    rows.append(dict(tag=tag, group=group_of(tag) or '', case_id=case['case_id'], video=case['video_id'], obj=case['object_id'], switch=case['switch_frame'], method=method, **got))
    return rows


def merged(rows):
    """{group: {method: {case_id: row}}} under the priority order; rounds outside the order stay out of the merged view."""
    rank = {t: i for i, t in enumerate(PRIORITY)}
    table = defaultdict(lambda: defaultdict(dict))
    for row in sorted((r for r in rows if r['group'] and r['tag'].rpartition('_')[0] in rank), key=lambda r: rank[r['tag'].rpartition('_')[0]]):
        table[row['group']][row['method']].setdefault(row['case_id'], row)
    return table


def summary(table, scan, out):
    flagged = {n for rows in scan.values() for n, v in rows.items() if v['shares']['0.97'] >= .25}
    level_rows, pair_rows = [], []
    for group, methods in table.items():
        for scope in ('all', 'clean'):
            def means(method, windows='jf'):
                return video_means((r['video'], r[windows]) for r in methods[method].values() if scope == 'all' or r['video'] not in flagged)
            for method in methods:
                got = means(method)
                if got:
                    level_rows.append((group, scope, method, num(np.mean(list(got.values())), 2), len(got)))
            for ref in REFERENCES:
                if ref not in methods:
                    continue
                base = means(ref)
                for method in methods:
                    if method != ref:
                        p = paired(means(method), base)
                        if p:
                            pair_rows.append((group, scope, method, ref, num(p['diff'], 2), num(p['low'], 2), num(p['high'], 2), p['videos'], p['better'], p['worse']))
    write(out / 'jf_levels.csv', ('group', 'videos_scope', 'method', 'jf', 'videos'), level_rows)
    write(out / 'jf_paired.csv', ('group', 'videos_scope', 'method', 'reference', 'diff', 'ci_low', 'ci_high', 'videos', 'better_by_1pt', 'worse_by_1pt'), pair_rows)


def windows(table, out):
    rows = []
    for group, methods in table.items():
        if 'target_ref' not in methods:
            continue
        for method in methods:
            for name in WINDOWS:
                a = video_means((r['video'], r[name]) for r in methods[method].values())
                b = video_means((r['video'], r[name]) for r in methods['target_ref'].values())
                p = paired(a, b) if method != 'target_ref' else None
                if a:
                    rows.append((group, method, name, num(np.mean(list(a.values())), 2), num(p['diff'], 2) if p else '', num(p['low'], 2) if p else '', num(p['high'], 2) if p else '', p['videos'] if p else len(a)))
    write(out / 'jf_windows.csv', ('group', 'method', 'window', 'jf', 'diff_from_base_state', 'ci_low', 'ci_high', 'videos'), rows)


def read_external(runs):
    rows = []
    for key in ('pumavos', 'm3vos'):
        for tag in (key, f'{key}2', f'{key}3', f'{key}4', f'{key}5'):
            for path in sorted(glob.glob(f'{runs}/unseen/{tag}/cases/*.json')):
                case = json.load(open(path))
                for name, row in case['switches'].items():
                    for method, score in row.items():
                        if isinstance(score, dict) and 'JF' in score and not method.endswith('_first30'):
                            first = row.get(f'{method}_first30') or {}
                            rows.append(dict(dataset=key, tag=tag, video=case['video'], obj=case['object'], name=name, switch=row.get('switch'), method=method, j=score['J'], f=score['F'], jf=score['JF'], n=score.get('n'), first30=first.get('JF')))
    return rows


def external(rows, out):
    write(out / 'external_case_scores.csv', ('dataset', 'tag', 'video', 'object', 'switch_name', 'switch_frame', 'method', 'j', 'f', 'jf', 'frames', 'jf_first30'),
          [(r['dataset'], r['tag'], r['video'], r['obj'], r['name'], r['switch'], r['method'], num(None if r['j'] is None else r['j'] * 100), num(None if r['f'] is None else r['f'] * 100), num(None if r['jf'] is None else r['jf'] * 100), r['n'], num(None if r['first30'] is None else r['first30'] * 100)) for r in rows])
    # The rounds share the Small and Base+ runs, so a later tag only fills methods the earlier ones lack, unless it is on the list of cases
    # with both fine-tuning rounds, which unseen_pair.py uses as the common set.
    table = defaultdict(dict)
    for r in sorted(rows, key=lambda r: (r['tag'].endswith('3'), r['tag'].endswith('4'), r['tag'].endswith('5'), r['tag'].endswith('2'))):
        table[(r['dataset'], r['video'], r['obj'], r['name'])].setdefault(r['method'], r)
    levels, pairs = [], []
    for key in ('pumavos', 'm3vos'):
        for scope, names in (('all switches', None), ('25', ('25',)), ('50', ('50',)), ('75', ('75',))):
            keys = [k for k, v in table.items() if k[0] == key and 'e2e3_ctx_f' in v and 'e2e_ctx_f' in v and (names is None or k[3] in names)]
            if not keys:
                continue
            methods = sorted({m for k in keys for m in table[k]})

            def means(method):
                return video_means((k[1], None if table[k].get(method, {}).get('jf') is None else table[k][method]['jf'] * 100) for k in keys if method in table[k])
            for method in methods:
                got = means(method)
                if got:
                    levels.append((key, scope, method, num(np.mean(list(got.values())), 2), len(got), len(keys)))
            for ref in ('base', 'legacy', 'mlp_context', 'e2e_ctx_f', 'small'):
                if ref in methods:
                    for method in methods:
                        if method != ref:
                            p = paired(means(method), means(ref))
                            if p:
                                pairs.append((key, scope, method, ref, num(p['diff'], 2), num(p['low'], 2), num(p['high'], 2), p['videos'], p['better'], p['worse']))
    write(out / 'external_levels.csv', ('dataset', 'switch', 'method', 'jf', 'videos', 'cases'), levels)
    write(out / 'external_paired.csv', ('dataset', 'switch', 'method', 'reference', 'diff', 'ci_low', 'ci_high', 'videos', 'better_by_1pt', 'worse_by_1pt'), pairs)
    return table


def paper_tables(folder, table, out):
    con = sqlite3.connect(f'file:{folder}/r2_epoch26/benchmark.sqlite?mode=ro', uri=True)
    rows = con.execute("select video, object_id, switch_name, switch_frame, start_frame, end_frame, baseline, run_id, j, jf, n_frames, failure_rate, switch_seconds, switch_gpu_mb from results where dataset='pumavos'").fetchall()
    write(out / 'paper_pumavos_cases.csv', ('video', 'object', 'switch_name', 'switch_frame', 'start_frame', 'end_frame', 'method', 'run', 'j', 'jf', 'frames', 'failure_rate', 'switch_seconds', 'switch_gpu_mb'),
          [(a, b, c, d, e, f, g, h, num(None if i is None else i * 100), num(None if j is None else j * 100), k, num(l), num(m, 5), num(n, 2)) for a, b, c, d, e, f, g, h, i, j, k, l, m, n in rows])
    con4090 = sqlite3.connect(f'file:{folder}/r2_epoch26_rtx4090/benchmark.sqlite?mode=ro', uri=True)
    vost = con4090.execute("select video, object_id, switch_name, switch_frame, start_frame, end_frame, baseline, run_id, j, jf, n_frames, failure_rate, switch_seconds, switch_gpu_mb from results where dataset='vost_val'").fetchall()
    write(out / 'paper_vost_cases.csv', ('video', 'object', 'switch_name', 'switch_frame', 'start_frame', 'end_frame', 'method', 'run', 'j', 'jf', 'frames', 'failure_rate', 'switch_seconds', 'switch_gpu_mb'),
          [(a, b, c, d, e, f, g, h, num(None if i is None else i * 100), num(None if j is None else j * 100), k, num(l), num(m, 5), num(n, 2)) for a, b, c, d, e, f, g, h, i, j, k, l, m, n in vost])

    paper = defaultdict(lambda: defaultdict(dict))
    meta = {}
    for video, obj, name, frame, start, end, base, run, j, jf, n, fail, sec, gpu in rows:
        paper[(video, obj, name)][base][run] = dict(jf=None if jf is None else jf * 100, n=n, sec=sec, gpu=gpu)
        meta[(video, obj, name)] = frame

    def paper_case(key, base):
        got = [r['jf'] for r in paper[key].get(base, {}).values() if r['jf'] is not None]
        return float(np.mean(got)) if got else None

    def review_case(key, method):
        row = table.get(('pumavos',) + key, {}).get(method)
        return None if row is None or row['jf'] is None else row['jf'] * 100

    keys = sorted(paper)
    agree = []
    for base, method in (('source_only', 'small'), ('full_replay', 'base'), ('original_last_visible', 'olv'), ('direct_state_copy', 'direct')):
        d = np.array([review_case(k, method) - paper_case(k, base) for k in keys])
        agree.append((PAPER[base], method, len(d), num(d.mean(), 3), num(np.median(d), 3), int((abs(d) < .1).sum()), int((abs(d) < 1).sum()), num(abs(d).max(), 2)))
    agree.append(('same switch frame', '', len(keys), '', '', sum(meta[k] == table[('pumavos',) + k]['small']['switch'] for k in keys), '', ''))
    agree.append(('same frame count', '', len(keys), '', '', sum(paper[k]['source_only'][1]['n'] == table[('pumavos',) + k]['small']['n'] for k in keys), '', ''))
    write(out / 'paper_evaluator_agreement.csv', ('paper_method', 'review_method', 'cases', 'mean_diff', 'median_diff', 'within_0.1', 'within_1', 'max_abs'), agree)

    levels = []
    for label, fn in [(PAPER[b], (lambda k, b=b: paper_case(k, b))) for b in PAPER] + [(m, (lambda k, m=m: review_case(k, m))) for m in REVIEW_FOR_PAPER]:
        cells = []
        for name in ('25', '50', '75'):
            got = video_means((k[0], fn(k)) for k in keys if k[2] == name)
            cells.append(float(np.mean(list(got.values()))))
        levels.append(('paper database' if label in PAPER.values() else 'design review', label, num(cells[0], 2), num(cells[1], 2), num(cells[2], 2), num(np.mean(cells), 2)))
    write(out / 'paper_pumavos_levels.csv', ('source', 'method', 'switch_25', 'switch_50', 'switch_75', 'mean_of_three'), levels)

    pairs = []
    for scope, names in (('50', ('50',)), ('25', ('25',)), ('75', ('75',)), ('all switches', ('25', '50', '75'))):
        for base in ('translator', 'source_only', 'full_replay', 'original_last_visible'):
            b = video_means((k[0], paper_case(k, base)) for k in keys if k[2] in names)
            for method in REVIEW_FOR_PAPER:
                a = video_means((k[0], review_case(k, method)) for k in keys if k[2] in names)
                p = paired(a, b)
                if p:
                    pairs.append((scope, method, PAPER[base], num(p['diff'], 2), num(p['low'], 2), num(p['high'], 2), p['videos'], p['better'], p['worse']))
    write(out / 'paper_pumavos_paired.csv', ('switch', 'method', 'paper_method', 'diff', 'ci_low', 'ci_high', 'videos', 'better_by_1pt', 'worse_by_1pt'), pairs)

    per_video = []
    for video in sorted({k[0] for k in keys}):
        selected = [k for k in keys if k[0] == video and k[2] == '50']
        row = [video]
        for fn in (lambda k: paper_case(k, 'translator'), lambda k: review_case(k, 'legacy'), lambda k: review_case(k, 'e2e_ctx_f'), lambda k: review_case(k, 'e2e3_ctx_f'), lambda k: review_case(k, 'e2e_ctx_pin7'), lambda k: review_case(k, 'e2e3_ctx_pin5'), lambda k: review_case(k, 'base'), lambda k: review_case(k, 'small')):
            row.append(num(np.mean([fn(k) for k in selected]), 2))
        per_video.append(row)
    write(out / 'paper_pumavos_video_50.csv', ('video', 'paper_translator', 'legacy', 'e2e_ctx', 'e2e3_ctx', 'e2e_ctx_pin7', 'e2e3_ctx_pin5', 'base', 'small'), per_video)

    cost = []
    for label, connection, dataset in (('L4', con, 'pumavos'), ('L4', con, 'vost_val'), ('RTX 4090', con4090, 'vost_val')):
        for base in PAPER:
            for name in ('25', '50', '75'):
                got = connection.execute('select switch_seconds, switch_gpu_mb from results where dataset=? and baseline=? and switch_name=? and switch_seconds is not null', (dataset, base, name)).fetchall()
                if got:
                    seconds = [g[0] for g in got]
                    cost.append((label, dataset, PAPER[base], name, len(got), num(np.mean(seconds), 5), num(np.median(seconds), 5), num(np.mean([g[1] for g in got]), 1)))
    write(out / 'paper_cost.csv', ('gpu', 'dataset', 'method', 'switch', 'rows', 'seconds_mean', 'seconds_median', 'gpu_mib_mean'), cost)


def training(runs, out):
    rows, history = [], []
    for path in sorted(glob.glob(f'{runs}/spatial/*/result.json')):
        name = Path(path).parent.name
        r = json.load(open(path))
        args = r.get('args', {})
        dev = r.get('dev') or {}
        rows.append((name, 'train_e2e' if 'init' in args else 'train_spatial', args.get('init', ''), args.get('updates', ''), args.get('batch', ''), args.get('lr', ''), r.get('parameters', ''), num(dev.get('all')) if isinstance(dev, dict) else '', num(r.get('seconds'), 0), json.dumps(args, ensure_ascii=False, sort_keys=True)))
        for item in r.get('history', []):
            if 'init' in args and isinstance(item, dict) and 'dev' in item:
                for scenario, values in item['dev'].items():
                    if isinstance(values, dict):
                        history.append((name, item.get('step'), scenario, *(num(values.get(k), 5) for k in ('feature', 'mask', 'score', 'state', 'iou'))))
    write(out / 'training_models.csv', ('model', 'trainer', 'init', 'updates', 'batch', 'lr', 'parameters', 'dev_all_mse', 'seconds', 'args'), rows)
    write(out / 'training_e2e_history.csv', ('model', 'step', 'scenario', 'feature', 'mask', 'score', 'state', 'iou'), history)
    pointers = []
    for path in sorted(glob.glob(f'{runs}/pointer/*.json')):
        r = json.load(open(path))
        pointers.append((Path(path).stem, json.dumps(r, ensure_ascii=False, sort_keys=True)[:4000]))
    write(out / 'pointer_experiments.csv', ('experiment', 'result_json'), pointers)


def cost(runs, out):
    data = json.load(open(f'{runs}/cost/cost.json'))
    seconds, peaks = defaultdict(list), defaultdict(list)
    for row in data:
        for k, v in row['seconds'].items():
            seconds[k].append(v)
        for k, v in row['peak_mb'].items():
            peaks[k].append(v)
    write(out / 'cost_summary.csv', ('method', 'cases', 'ms_mean', 'ms_median', 'peak_mib_mean'), [(k, len(v), num(np.mean(v) * 1000, 2), num(np.median(v) * 1000, 2), num(np.mean(peaks[k]), 1)) for k, v in seconds.items()])


def scenes(scan, out):
    rows = []
    for group, videos in scan.items():
        rows.append((group, len(videos), *(sum(v['shares'][level] >= .25 for v in videos.values()) for level in ('0.93', '0.95', '0.97', '0.98'))))
    write(out / 'scene_overlap.csv', ('group', 'videos', 'shared_at_0.93', 'shared_at_0.95', 'shared_at_0.97', 'shared_at_0.98'), rows)
    rows = []
    for group, videos in scan.items():
        for name, v in sorted(videos.items()):
            if v['shares']['0.97'] >= .25:
                rows.append((group, name, v['shares']['0.97'], v.get('match'), v.get('match_group')))
    write(out / 'scene_overlap_videos.csv', ('group', 'video', 'share_at_0.97', 'matching_fit_video', 'match_group'), rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True)
    parser.add_argument('--paper')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jf(args.runs)
    keys = ('tag', 'group', 'case_id', 'video', 'obj', 'switch', 'method', 'n', 'j', 'f', 'jf', *WINDOWS)
    write(out / 'jf_case_scores.csv', keys, [[num(r[k]) if isinstance(r[k], float) else ('' if r[k] is None else r[k]) for k in keys] for r in rows])
    table = merged(rows)
    scan = json.load(open(f'{args.runs}/analysis/dup_scan.json'))
    summary(table, scan, out)
    windows(table, out)
    reviewed = external(read_external(args.runs), out)
    if args.paper:  # the paper databases are not part of test14; paper_*.csv stay as published here without them
        paper_tables(args.paper, reviewed, out)
    training(args.runs, out)
    cost(args.runs, out)
    scenes(scan, out)


if __name__ == '__main__':
    main()
