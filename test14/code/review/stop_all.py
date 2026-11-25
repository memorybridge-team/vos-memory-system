#!/usr/bin/env python3
"""Stops this review's experiments by exact PID. Without --go it only lists them.

A process is selected only when its working directory is the review root and its arguments are a pipeline script, an experiment script
or the governor. Supervisors go first so nothing restarts, the governor last so no experiment stays paused. Finished cases stay on disk.
"""
import os
import re
import signal
import sys
import time

ROOT = '/workspace/cmmt-design-review-20261104'
PIPELINE = re.compile(r'(^|/)code/pipeline\d*\.sh$')
EXPERIMENT = re.compile(r'^code/(handoff_eval|more_eval|jf_eval|unseen_eval|unseen_more|train_spatial|train_e2e)\.py$')


def rows():
    found = dict(supervisor=[], experiment=[], governor=[])
    for entry in os.scandir('/proc'):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as handle:
                argv = [a.decode(errors='replace') for a in handle.read().split(b'\0') if a]
            cwd = os.readlink(f'/proc/{pid}/cwd')
        except OSError:
            continue
        if len(argv) < 2 or cwd != ROOT:
            continue
        name = os.path.basename(argv[0])
        if name == 'bash' and PIPELINE.search(argv[1]):
            found['supervisor'].append((pid, ' '.join(argv)[:90]))
        elif name.startswith('python') and EXPERIMENT.match(argv[1]):
            found['experiment'].append((pid, ' '.join(argv)[:90]))
        elif name.startswith('python') and argv[1] == 'code/governor.py':
            found['governor'].append((pid, ' '.join(argv)[:90]))
    return found


def alive(pid):
    try:
        with open(f'/proc/{pid}/stat') as handle:
            return handle.read().rsplit(')', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


def main():
    found = rows()
    for kind, items in found.items():
        print(kind, len(items))
        for pid, text in sorted(items):
            print(f'  {pid} {text}')
    if '--go' not in sys.argv:
        return
    for pid, _ in found['supervisor']:
        os.kill(pid, signal.SIGTERM)
    time.sleep(2)
    found = rows()
    for pid, _ in found['supervisor']:  # a loop can start another step between the scan and the signal
        os.kill(pid, signal.SIGKILL)
    time.sleep(1)
    experiments = [pid for pid, _ in rows()['experiment']]
    for pid in experiments:
        os.kill(pid, signal.SIGCONT)
        os.kill(pid, signal.SIGTERM)
    time.sleep(8)
    for pid in experiments:
        if alive(pid):
            os.kill(pid, signal.SIGKILL)
    time.sleep(2)
    for pid, _ in rows()['governor']:
        os.kill(pid, signal.SIGTERM)
    time.sleep(2)
    print('after stop:', {k: len(v) for k, v in rows().items()})
    print('STOP_DONE')


if __name__ == '__main__':
    main()
