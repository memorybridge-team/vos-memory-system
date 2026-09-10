#!/usr/bin/env python3
"""Run the complete four-model matrix on a small RunPod disk, one shard at a time."""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datasets import discover
from merge_results import merge_spec_results
from prepare_local import stage_dataset


MODELS = ("tiny", "small", "plus", "large")
DATASETS = (
    {"name": "davis2017", "split": "val"},
    {"name": "lvos", "version": "v1", "split": "val"},
    {"name": "mosev2", "split": "train"},
    {"name": "mosev2", "split": "valid"},
)


def log(report, message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"- {stamp} — {message}\n"
    print(line, end="", flush=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("a") as stream:
        stream.write("\n" + line)


def command(args, *, capture=False):
    return subprocess.run(args, check=True, text=True,
                          capture_output=capture)


class RunPod:
    def __init__(self, host, port, key, report):
        self.host = host
        self.port = str(port)
        self.key = str(Path(key).expanduser())
        self.report = report
        self.ssh_base = [
            "ssh", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=20",
            "-i", self.key, "-p", self.port, self.host,
        ]
        self.rsync_ssh = (
            f"ssh -o BatchMode=yes -o IdentitiesOnly=yes -i {self.key} "
            f"-o ServerAliveInterval=30 -o ServerAliveCountMax=20 -p {self.port}"
        )

    def ssh(self, remote_command, label):
        log(self.report, f"RunPod command start: {label}")
        result = subprocess.run(self.ssh_base + [remote_command], text=True,
                                capture_output=True)
        if result.returncode:
            log(self.report, f"RunPod command failed: {label}: {result.stderr[-1200:]}")
            raise subprocess.CalledProcessError(result.returncode, result.args,
                                                result.stdout, result.stderr)
        if result.stdout.strip():
            log(self.report, f"RunPod command complete: {label}: {result.stdout[-500:].strip()}")
        else:
            log(self.report, f"RunPod command complete: {label}")
        return result.stdout

    def sync_to(self, source, destination, label):
        log(self.report, f"Transfer start: {label}")
        source_path = Path(source).resolve()
        args = ["rsync", "-a", "--partial"]
        if source_path.is_dir():
            args.append("--delete")
            source_arg = f"{source_path}/"
            destination_arg = f"{self.host}:{destination.rstrip('/')}/"
        else:
            source_arg = str(source_path)
            destination_arg = f"{self.host}:{destination}"
        command([*args, "-e", self.rsync_ssh, source_arg, destination_arg])
        log(self.report, f"Transfer complete: {label}")

    def sync_from(self, source, destination, label):
        Path(destination).mkdir(parents=True, exist_ok=True)
        log(self.report, f"Result retrieval start: {label}")
        command([
            "rsync", "-a", "--partial",
            "--exclude", "*/videos/*/cold/masks/",
            "--exclude", "*/videos/*/full_memory/masks/",
            "-e", self.rsync_ssh, f"{self.host}:{source.rstrip('/')}/",
            f"{Path(destination).resolve()}/",
        ])
        log(self.report, f"Result retrieval complete: {label}")

    def verify_run_status(self, spec):
        name = json.dumps(spec["name"])
        split = json.dumps(spec["split"])
        code = (
            "import json,pathlib; "
            "root=pathlib.Path('/workspace/results'); "
            f"items=[p for p in root.glob('*/run.json') if (lambda d: "
            f"d.get('config',{{}}).get('datasets',[{{}}])[0].get('name')=={name} and "
            f"d.get('config',{{}}).get('datasets',[{{}}])[0].get('split')=={split})"
            "(json.loads(p.read_text()))]; "
            "p=max(items,key=lambda x:x.stat().st_mtime_ns).parent; "
            "s=json.loads((p/'run-status.json').read_text()); "
            "print(json.dumps(s)); "
            "raise SystemExit(1 if s.get('failed') or s.get('failed_combinations') else 0)"
        )
        remote = f"/workspace/venv/bin/python -c {shlex.quote(code)}"
        return self.ssh(remote, f"verify {spec['name']}/{spec['split']}")

    def cleanup_dataset(self, spec, remote_data):
        """Free exact, already-retrieved dataset data/results on the small pod disk."""
        name = json.dumps(spec["name"])
        split = json.dumps(spec["split"])
        code = (
            "import json,pathlib,shutil; "
            "root=pathlib.Path('/workspace/results'); "
            f"items=[p for p in root.glob('*/run.json') if (lambda d: "
            f"d.get('config',{{}}).get('datasets',[{{}}])[0].get('name')=={name} and "
            f"d.get('config',{{}}).get('datasets',[{{}}])[0].get('split')=={split})"
            "(json.loads(p.read_text()))]; "
            "[shutil.rmtree(p.parent) for p in items if p.parent.exists()]; "
            f"shutil.rmtree({json.dumps(remote_data)}, ignore_errors=True); "
            "print('cleaned', len(items))"
        )
        remote = f"/workspace/venv/bin/python -c {shlex.quote(code)}"
        return self.ssh(remote, f"cleanup {spec['name']}/{spec['split']}")


def spec_source(spec, cache):
    return Path(cache) / "datasets" / spec["name"] / spec["split"]


def video_size(video):
    return sum(path.stat().st_size for path in [*video.frames, *video.masks.values()])


def chunks(videos, max_bytes):
    current, current_size = [], 0
    for video in videos:
        size = video_size(video)
        if current and current_size + size > max_bytes:
            yield current
            current, current_size = [], 0
        current.append(video.name)
        current_size += size
    if current:
        yield current


def write_config(path, spec, remote_root, output_root, models, run_tag):
    dataset = {"name": spec["name"], "split": spec["split"], "root": remote_root}
    if spec["name"] == "lvos":
        dataset["version"] = "v1"
    config = {
        "cache": "/workspace/cache",
        "output": output_root,
        "sam2_repo": "/workspace/appendix_1/vendor/sam2",
        "models": list(models),
        "conditions": ["cold", "full_memory"],
        "datasets": [dataset],
        "device": "cuda",
        "precision": "bfloat16",
        "seed": 42,
        "save_masks": spec["name"] == "mosev2" and spec["split"] == "valid",
        "offload_video_to_cpu": True,
        "offload_state_to_cpu": True,
        "max_videos": None,
        "bootstrap_samples": 10000,
        "checkpoints": {},
        "run_tag": run_tag,
    }
    path.write_text(json.dumps(config, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="root@PUBLIC_IP")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--key", default="~/.ssh/runpod_ed25519")
    parser.add_argument("--cache", type=Path, default=HERE / "local_data" / "cache")
    parser.add_argument("--report", type=Path, default=HERE / "RUNPOD_PROGRESS.md")
    parser.add_argument("--max-shard-gb", type=float, default=7.0)
    parser.add_argument("--valid-shard-gb", type=float, default=2.0,
                        help="smaller input shard for MOSEv2 valid mask submissions")
    parser.add_argument("--only", choices=("davis2017/val", "lvos/val",
                                             "mosev2/train", "mosev2/valid"),
                        action="append")
    args = parser.parse_args()
    if args.max_shard_gb <= 0:
        raise SystemExit("--max-shard-gb must be positive")
    if args.valid_shard_gb <= 0:
        raise SystemExit("--valid-shard-gb must be positive")

    report = args.report.resolve()
    pod = RunPod(args.host, args.port, args.key, report)
    cache = args.cache.resolve()
    staging_root = HERE / "local_data" / "runpod_staging"
    config_path = HERE / "benchmark.runpod.batch.json"
    local_results = HERE / "runpod_results"
    state_path = HERE / "local_data" / "runpod_batch_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"completed": []}
    completed = set(state.get("completed", []))
    selected_specs = set(args.only or [])

    for spec in DATASETS:
        spec_key = f"{spec['name']}/{spec['split']}"
        if selected_specs and spec_key not in selected_specs:
            continue
        source = spec_source(spec, cache)
        videos = sorted(discover(spec, source), key=lambda video: video.name)
        shard_gb = (min(args.max_shard_gb, args.valid_shard_gb)
                    if spec["name"] == "mosev2" and spec["split"] == "valid"
                    else args.max_shard_gb)
        shard_list = list(chunks(videos, int(shard_gb * 1024 ** 3)))
        remote_data = f"/workspace/data/{spec['name']}_{spec['split']}"
        output_root = "/workspace/results"
        valid_submissions = spec["name"] == "mosev2" and spec["split"] == "valid"
        unit_models = tuple(MODELS) if valid_submissions else (None,)
        log(report, f"{spec_key}: {len(videos)} videos, {len(shard_list)} shards, "
            f"models={','.join(MODELS)}, valid_shard_gb={shard_gb:g}")

        for index, names in enumerate(shard_list, start=1):
            shard_key = f"{spec_key}#shard-{index:04d}"
            pending_models = []
            for model in unit_models:
                unit_key = f"{shard_key}#model-{model}" if valid_submissions else shard_key
                if unit_key not in completed:
                    pending_models.append((model, unit_key))
            if not pending_models:
                log(report, f"skip completed {shard_key}")
                continue
            if staging_root.exists():
                shutil.rmtree(staging_root)
            staged = staging_root / f"{spec['name']}_{spec['split']}"
            stage_dataset(spec, source, staged, names)
            for model, unit_key in pending_models:
                models = [model] if valid_submissions else list(MODELS)
                run_tag = unit_key.replace("/", "_").replace("#", "_")
                write_config(config_path, spec, remote_data, output_root, models, run_tag)
                pod.sync_to(config_path, "/workspace/appendix_1/benchmark.runpod.batch.json",
                            f"{unit_key} config")
                pod.sync_to(staged, remote_data, unit_key)
                log(report, f"benchmark start: {unit_key} ({len(names)} videos, "
                    f"models={','.join(models)})")
                log_model = model or "all"
                remote_log = f"/workspace/logs/{spec['name']}_{spec['split']}_{index:04d}_{log_model}.log"
                pod.ssh(
                    "mkdir -p /workspace/logs && "
                    f"cd /workspace/appendix_1 && "
                    f"/workspace/venv/bin/python benchmark.py "
                    f"--config benchmark.runpod.batch.json run > {remote_log} 2>&1",
                    f"benchmark {unit_key}",
                )
                pod.verify_run_status(spec)
                pod.sync_from("/workspace/results", local_results, f"{unit_key} results")
                completed.add(unit_key)
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps({"completed": sorted(completed)}, indent=2) + "\n")
                log(report, f"benchmark complete: {unit_key}")
                pod.cleanup_dataset(spec, remote_data)
            shutil.rmtree(staging_root, ignore_errors=True)

        summary = merge_spec_results(local_results, spec)
        log(report, f"local merged summary {spec_key}: "
            f"completed={summary['completed_video_jobs']} "
            f"failed={summary['failed_video_jobs']} "
            f"missing={summary['missing_video_jobs']}")

    log(report, "all requested RunPod shards completed")


if __name__ == "__main__":
    main()
