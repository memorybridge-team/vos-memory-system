#!/usr/bin/env python3
"""Load governor for the design review: keeps GPU use and fan noise down.

At most N experiment processes run per GPU, highest priority first. The others are paused
with SIGSTOP (GPU memory stays allocated) and continue by themselves when a slot frees up,
so the pipelines need no attention. Tune while it runs:
    echo 2  > runs/control/max_per_gpu     # 0 pauses everything
    echo 60 > runs/control/duty            # percent of each 10 s cycle the slots may run
    echo 3  > runs/control/max_per_gpu.0   # overrides max_per_gpu on GPU 0 only
    echo 100 > runs/control/train_duty.0   # training share of the cycle on GPU 0 only
SIGTERM resumes every paused process and exits.
"""
import os
import re
import signal
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/cmmt-design-review-20261104')
CONTROL = ROOT / 'runs/control'
SCRIPT = re.compile(r'code/(handoff_eval|more_eval|jf_eval|unseen_eval|unseen_more|train_spatial|train_e2e)\.py')
TAG = re.compile(r'--tag (\S+)')
ORDER = ('t2_val', 't5_mose', 't2_mose', 't6_mose', 't5_val', 't6_val', 't4_mose', 't4_val', 'cand_mose', 'cand_val',
         'fp32_val', 'fp32_mose', 'pumavos2', 'm3vos2', 'train', 't3_val', 't3_mose')
PERIOD = 10
TRAIN_DUTY = 50  # training alone would hold a GPU at 100 %


def number(name, default):
    try:
        return int((CONTROL / name).read_text().strip())
    except (OSError, ValueError):
        return default


def order():
    try:
        return tuple((CONTROL / 'order').read_text().split())
    except OSError:
        return ORDER


def experiments():
    ranking = order()
    rows = []
    for entry in os.scandir('/proc'):
        if not entry.name.isdigit():
            continue
        try:
            with open(f'/proc/{entry.name}/cmdline', 'rb') as handle:
                command = handle.read().split(b'\0')
            if not os.path.basename(command[0].decode(errors='replace')).startswith('python'):
                continue
            text = b' '.join(command).decode(errors='replace')
            if not SCRIPT.search(text):
                continue
            with open(f'/proc/{entry.name}/environ', 'rb') as handle:
                environment = handle.read().split(b'\0')
            with open(f'/proc/{entry.name}/stat') as handle:
                state = handle.read().rsplit(')', 1)[1].split()[0]
        except (OSError, IndexError):
            continue
        gpu = next((v[21:].decode() for v in environment if v.startswith(b'CUDA_VISIBLE_DEVICES=')), '?')
        match = TAG.search(text)
        tag = 'train' if 'train_spatial' in text else (match.group(1) if match else '?')
        rank = ranking.index(tag) if tag in ranking else len(ranking)
        rows.append(dict(pid=int(entry.name), gpu=gpu, tag=tag, rank=rank, state=state))
    return rows


def apply(pid, state, run):
    try:
        if run and state == 'T':
            os.kill(pid, signal.SIGCONT)
        elif not run and state != 'T':
            os.kill(pid, signal.SIGSTOP)
    except ProcessLookupError:
        pass


def resume_all(*_):
    for row in experiments():
        apply(row['pid'], row['state'], True)
    print(time.strftime('%F %T'), 'governor stopped, everything resumed', flush=True)
    sys.exit(0)


def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    if not (CONTROL / 'max_per_gpu').exists():
        (CONTROL / 'max_per_gpu').write_text('1\n')
    if not (CONTROL / 'duty').exists():
        (CONTROL / 'duty').write_text('100\n')
    signal.signal(signal.SIGTERM, resume_all)
    signal.signal(signal.SIGINT, resume_all)
    print(time.strftime('%F %T'), 'governor started pid', os.getpid(), flush=True)
    previous = None
    try:
        while True:
            limit, duty = number('max_per_gpu', 1), number('duty', 100)
            phase = (time.time() % PERIOD) / PERIOD * 100
            rows = experiments()
            active = []
            for gpu in sorted({r['gpu'] for r in rows}):
                group = sorted((r for r in rows if r['gpu'] == gpu), key=lambda r: (r['rank'], r['pid']))
                slots = number(f'max_per_gpu.{gpu}', limit)
                train_share = number(f'train_duty.{gpu}', TRAIN_DUTY)
                for index, row in enumerate(group):
                    allowed = index < slots
                    if allowed:
                        active.append(f"gpu{gpu}:{row['tag']}")
                    share = min(duty, train_share) if row['tag'] == 'train' or row['tag'].startswith('e2e') else duty
                    apply(row['pid'], row['state'], allowed and phase < share)
            now = (limit, duty, tuple(active))
            if now != previous:
                print(time.strftime('%F %T'), f'limit {limit} duty {duty} running {" ".join(active) or "none"} of {len(rows)}', flush=True)
                previous = now
            if not rows and all((ROOT / 'runs/reports' / name).exists() for name in ('ALL_DONE', 'ALL_DONE2', 'ALL_DONE3', 'ALL_DONE4', 'ALL_DONE5', 'ALL_DONE6')):
                print(time.strftime('%F %T'), 'all experiments finished', flush=True)
                return
            time.sleep(1)
    finally:
        for row in experiments():
            apply(row['pid'], row['state'], True)


if __name__ == '__main__':
    main()
