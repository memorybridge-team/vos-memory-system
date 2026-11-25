#!/usr/bin/env python3
"""Pre-run checks of memory-resume-20261120 (plan section 5), read from finished case files.

  python3 code/checks.py --tag smoke_pumavos --tag smoke_m3vos \\
      --repeat smoke_pumavos=smoke_pumavos_r --repeat smoke_m3vos=smoke_m3vos_r --out checks/smoke.json

  1  resume equals Base+ alone on every scored frame (J and F, exact).
  2  every continuation that should hold Base+'s own original-prompt record holds it bit for bit.
  3  small, base, olv, restart_p0 (orig), lastvis, paper and conv_strip_vm reproduce earlier runs (same case, same switch).
  4  records read at s+1: resume and translators read 1 conditioning record, min(6, s-start) recent records and
     min(15, s-start) pointers; restarts read no recent record or pointer and only their prompt frames.
  5  prompt frames: resume and translators none; restart_p0 {start}; restart_mask {lv_B}; restart_p0_mask {start, lv_B};
     olv {start, lv_S}; lastvis {lv_S}. The continuation guard refuses every other past frame.
  6  a repeated run gives identical per-frame values.
"""
import argparse
import glob
import json
from pathlib import Path

E = Path('/workspace/memory-resume-20261120/runs/unseen')
LT = Path('/workspace/linear-transfer-large-20261108/vos/runs/unseen')
SB = Path('/workspace/linear-transfer-large-20261108/runs/sb-linear-20261113/runs/unseen')
DR = Path('/workspace/cmmt-design-review-20261104/runs/unseen')
STATE = ('resume', 'paper', 'conv_strip_vm')
OWN_P0 = ('resume', 'restart_p0', 'restart_p0_mask', 'olv')


def cases(folder):
    out = {}
    for f in glob.glob(str(folder / 'cases' / '*.json')):
        d = json.loads(Path(f).read_text())
        for label, row in d['switches'].items():
            out[(d['video'], d['object'], label)] = (d, row)
    return out


def references(dataset):
    """method -> list of (source name, {case key: JF}) from earlier runs of the same evaluator rules."""
    lt, dr = ('base_pumavos', 'pumavos') if dataset == 'pumavos' else ('base_m3vos', 'm3vos')
    def jf(folder, name):
        got = {}
        for key, (_, row) in cases(folder).items():
            if isinstance(row.get(name), dict) and row[name].get('JF') is not None:
                got[key] = row[name]['JF']
        return got
    return {
        'base': [(f'{lt}:base', jf(LT / lt, 'base')), (f'{dr}2:base', jf(DR / f'{dr}2', 'base'))],
        'small': [(f'{lt}:small', jf(LT / lt, 'small'))],
        'paper': [(f'{lt}:paper', jf(LT / lt, 'paper'))],
        'conv_strip_vm': [(f'{lt}_strip:conv_strip_vm', jf(SB / f'{lt}_strip', 'conv_strip_vm'))],
        'restart_p0': [(f'{dr}2:orig', jf(DR / f'{dr}2', 'orig'))],
        'lastvis': [(f'{dr}2:lastvis', jf(DR / f'{dr}2', 'lastvis'))],
        'olv': [(f'{dr}2:olv', jf(DR / f'{dr}2', 'olv')), (f'{dr}:olv', jf(DR / dr, 'olv'))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', action='append', required=True)
    ap.add_argument('--repeat', action='append', default=[], help='original=repeat tag pairs')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    report = dict(tags=args.tag, checks={})
    fails = {k: [] for k in ('1_resume_equals_base', '2_own_prompt_record', '4_records_read', '5_prompt_frames', '6_repeat')}
    repro = {}
    for tag in args.tag:
        found = cases(E / tag)
        dataset = next(iter(found.values()))[0]['dataset']
        refs = references(dataset)
        for key, (d, row) in sorted(found.items()):
            start, s = d['start'], row['switch']
            lv_b, lv_s = row['last_visible_base'], row['last_visible_small']
            if row.get('resume_max_abs_diff_vs_base', 1.0) != 0.0:
                fails['1_resume_equals_base'].append([tag, *key, row.get('resume_max_abs_diff_vs_base')])
            for m in OWN_P0 + ('restart_mask', 'lastvis'):
                read = row['reads'].get(m)
                if read is None:
                    continue
                own = m in OWN_P0 or read['cond'] == [start]
                if own and (read.get('p0_spatial_diff') != 0.0 or read.get('p0_pointer_diff') != 0.0):
                    fails['2_own_prompt_record'].append([tag, *key, m, read.get('p0_spatial_diff'), read.get('p0_pointer_diff')])
            expect_prompts = dict(resume=[], paper=[], conv_strip_vm=[], restart_p0=[start], restart_mask=[lv_b],
                                  restart_p0_mask=sorted({start, lv_b}), olv=sorted({start, lv_s}), lastvis=[lv_s])
            for m, want in expect_prompts.items():
                if m in row['prompts'] and sorted(row['prompts'][m]) != want:
                    fails['5_prompt_frames'].append([tag, *key, m, row['prompts'][m], want])
                read = row['reads'].get(m)
                if read is None:
                    continue
                if m in STATE:
                    ok = read['cond'] == [start] and read['recent6'] == min(6, s - start) and read['ptr15'] == min(15, s - start)
                else:
                    ok = read['cond'] == want and read['recent6'] == 0 and read['ptr15'] == 0
                if not ok or read['at'] != s + 1:
                    fails['4_records_read'].append([tag, *key, m, read])
            for m, sources in refs.items():
                if not isinstance(row.get(m), dict) or row[m].get('JF') is None:
                    continue
                for source, values in sources:
                    if key not in values:
                        continue
                    r = repro.setdefault(f'{dataset}:{m} vs {source}', dict(n=0, exact=0, max_abs=0.0, over_0_5=[]))
                    diff = abs(row[m]['JF'] - values[key])
                    r['n'] += 1
                    r['exact'] += diff < 1e-12
                    r['max_abs'] = max(r['max_abs'], diff)
                    if diff > 0.005:
                        r['over_0_5'].append([*key, round(100 * row[m]['JF'], 2), round(100 * values[key], 2)])
        report.setdefault('cases', {})[tag] = len(found)
    for pair in args.repeat:
        a, b = pair.split('=')
        A, B = cases(E / a), cases(E / b)
        common = sorted(set(A) & set(B))
        for key in common:
            ra, rb = A[key][1], B[key][1]
            if ra['trace'] != rb['trace'] or ra['reads'] != rb['reads']:
                fails['6_repeat'].append([a, b, *key])
        report.setdefault('repeat_compared', {})[pair] = len(common)
    for name, bad in fails.items():
        report['checks'][name] = dict(passed=not bad, failures=bad[:20], n_failures=len(bad))
    report['checks']['3_reproduction'] = repro
    Path(args.out).write_text(json.dumps(report, indent=1))
    for name, value in report['checks'].items():
        if name == '3_reproduction':
            for k, r in value.items():
                print(f"  3 {k}: {r['exact']}/{r['n']} exact, max |diff| {100 * r['max_abs']:.2f}, cases over 0.5: {len(r['over_0_5'])}")
        else:
            print(f"  {name}: {'PASS' if value['passed'] else 'FAIL'} ({value['n_failures']} failures)")
    print('cases', report.get('cases'), 'repeat', report.get('repeat_compared'))


if __name__ == '__main__':
    main()
