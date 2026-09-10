#!/usr/bin/env python3
"""Watch the active LVOS RunPod job, recover it, then run the MOSE batches."""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from merge_results import merge_spec_results


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(report, message):
    line = f"- {stamp()} — {message}\n"
    print(line, end="", flush=True)
    with report.open("a") as stream:
        stream.write("\n" + line)


def ssh_args(host, port, key):
    return [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=20",
        "-i", str(Path(key).expanduser()), "-p", str(port), host,
    ]


def run_remote(args, command):
    return subprocess.run(args + [command], text=True, capture_output=True,
                          check=False)


def active_status(args):
    command = (
        "ps -eo pid,etime,%cpu,cmd | "
        "grep -E 'benchmark.py.*--config benchmark.runpod.batch.json run' | "
        "grep -v grep || true"
    )
    result = run_remote(args, command)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def latest_lvos_run(args):
    code = r'''import json
from pathlib import Path
root = Path('/workspace/results')
items = []
for path in root.glob('*/run.json'):
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    datasets = data.get('config', {}).get('datasets', [])
    if not any(x.get('name') == 'lvos' and x.get('split') == 'val' for x in datasets):
        continue
    status = path.parent / 'run-status.json'
    if status.is_file():
        items.append((path.stat().st_mtime_ns, path.parent, json.loads(status.read_text())))
if items:
    _, path, status = max(items)
    print(json.dumps({'root': str(path), 'status': status}))
'''
    command = f"/workspace/venv/bin/python -c {shlex.quote(code)}"
    result = run_remote(args, command)
    if result.returncode or not result.stdout.strip():
        return None, result.stderr.strip() or result.stdout.strip()
    try:
        return json.loads(result.stdout.strip().splitlines()[-1]), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def update_state(state_path, key):
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"completed": []}
    completed = set(state.get("completed", []))
    completed.add(key)
    state_path.write_text(json.dumps({"completed": sorted(completed)}, indent=2) + "\n")


def cleanup_lvos(ssh):
    code = r'''import json, pathlib, shutil
root = pathlib.Path('/workspace/results')
items = []
for path in root.glob('*/run.json'):
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    datasets = data.get('config', {}).get('datasets', [])
    if any(x.get('name') == 'lvos' and x.get('split') == 'val' for x in datasets):
        items.append(path.parent)
for path in items:
    shutil.rmtree(path, ignore_errors=True)
shutil.rmtree('/workspace/data/lvos_val', ignore_errors=True)
print('cleaned', len(items))
'''
    result = run_remote(ssh, f"/workspace/venv/bin/python -c {shlex.quote(code)}")
    if result.returncode:
        raise RuntimeError(result.stderr[-1200:])


def resume_lvos(ssh, report):
    remote_log = "/workspace/logs/lvos_val_recovery.log"
    command = (
        "mkdir -p /workspace/logs && cd /workspace/appendix_1 && "
        "nohup /workspace/venv/bin/python benchmark.py "
        f"--config benchmark.runpod.batch.json run >> {remote_log} 2>&1 "
        "< /dev/null >/dev/null 2>&1 & echo RESUME_PID=$!"
    )
    log(report, "resuming incomplete LVOS run from saved per-video state")
    result = run_remote(ssh, command)
    if result.returncode:
        raise RuntimeError(result.stderr[-1200:] or "LVOS resume command failed")
    log(report, f"LVOS resume launched in background: {result.stdout.strip()[-200:]}")


def recover_lvos(cli, ssh, report, local_results, state_path):
    info, error = latest_lvos_run(ssh)
    if error:
        raise RuntimeError(f"could not locate LVOS run: {error}")
    status = info["status"]
    expected = 50 * 4
    if (status.get("completed", 0) != expected or status.get("failed", 0) or
            status.get("failed_combinations", 0)):
        raise RuntimeError(f"LVOS run incomplete: {status}")
    local_results.mkdir(parents=True, exist_ok=True)
    rsync_ssh = (
        f"ssh -o BatchMode=yes -o IdentitiesOnly=yes -i {Path(cli.key).expanduser()} "
        f"-o ServerAliveInterval=30 -o ServerAliveCountMax=20 -p {cli.port}"
    )
    command = [
        "rsync", "-a", "--partial",
        "--exclude", "*/videos/*/cold/masks/",
        "--exclude", "*/videos/*/full_memory/masks/",
        "-e", rsync_ssh, f"{cli.host}:/workspace/results/",
        f"{local_results.resolve()}/",
    ]
    subprocess.run(command, check=True)
    local_root = local_results / Path(info["root"]).name
    local_status = local_root / "run-status.json"
    if not local_status.is_file():
        raise RuntimeError(f"retrieved LVOS status is missing: {local_status}")
    local_status_data = json.loads(local_status.read_text())
    if local_status_data != status:
        raise RuntimeError("retrieved LVOS status differs from remote status")
    update_state(state_path, "lvos/val#shard-0001")
    summary = merge_spec_results(local_results, {
        "name": "lvos", "version": "v1", "split": "val",
    })
    if not summary["complete"]:
        raise RuntimeError(f"local LVOS merge is incomplete: {summary}")
    cleanup_lvos(ssh)
    log(report, f"LVOS recovered: completed={status['completed']}, failed={status['failed']}; "
        "local summary complete; exact remote inputs/results cleaned")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="root@PUBLIC_IP")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--key", default="~/.ssh/runpod_ed25519")
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    root = args.root.resolve()
    report = root / "appendix_1" / "RUNPOD_PROGRESS.md"
    local_results = root / "appendix_1" / "runpod_results"
    state_path = root / "appendix_1" / "local_data" / "runpod_batch_state.json"
    ssh = ssh_args(args.host, args.port, args.key)

    while True:
        code, output, error = active_status(ssh)
        if code:
            log(report, f"monitor SSH error; retrying in 300s: {error[-500:]}")
        elif output:
            compact = output.replace("\n", " | ")
            log(report, f"monitor: LVOS benchmark active: {compact[:700]}")
        else:
            log(report, "monitor: LVOS benchmark process ended; beginning recovery")
            break
        time.sleep(300)

    while True:
        try:
            recover_lvos(args, ssh, report, local_results, state_path)
            break
        except Exception as exc:
            log(report, f"LVOS recovery not yet verified; retrying in 300s: {exc}")
            info, _ = latest_lvos_run(ssh)
            status = info.get("status", {}) if info else {}
            incomplete = (not info or status.get("completed", 0) != 50 * 4 or
                           status.get("failed", 0) or status.get("failed_combinations", 0))
            data_check = run_remote(ssh, "test -d /workspace/data/lvos_val && echo DATA_PRESENT")
            if incomplete and data_check.returncode == 0 and "DATA_PRESENT" in data_check.stdout:
                try:
                    # A transient SSH failure can leave the original remote job
                    # alive.  Never launch a second benchmark until this check
                    # proves that no matching process exists.
                    active_code, active_output, active_error = active_status(ssh)
                    if active_code:
                        raise RuntimeError(
                            f"could not verify remote activity before resume: "
                            f"{active_error[-500:]}"
                        )
                    if active_output:
                        log(report, "remote LVOS process still active; skipping duplicate resume")
                        continue
                    resume_lvos(ssh, report)
                    continue
                except Exception as resume_error:
                    log(report, f"LVOS resume failed; retrying in 300s: {resume_error}")
            time.sleep(300)

    command = [
        sys.executable, "appendix_1/runpod_batch.py",
        "--host", args.host, "--port", str(args.port), "--key", str(args.key),
        "--max-shard-gb", "7", "--valid-shard-gb", "2",
        "--only", "mosev2/train", "--only", "mosev2/valid",
    ]
    log(report, "starting MOSEv2 train and valid RunPod batches")
    result = subprocess.run(command, cwd=root, text=True)
    if result.returncode:
        log(report, f"MOSEv2 batch runner exited with code {result.returncode}")
        raise SystemExit(result.returncode)
    log(report, "MOSEv2 train and valid RunPod batches completed")


if __name__ == "__main__":
    main()
