"""Merge sharded RunPod result roots into one locally summarized run."""

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from protocol import PROTOCOL_NAME
from summarize import summarize_run


def _matches(run_json, spec):
    datasets = run_json.get("config", {}).get("datasets", [])
    return (run_json.get("protocol") == PROTOCOL_NAME and
            any(item.get("name") == spec["name"] and
                item.get("split") == spec["split"] for item in datasets))


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"unsafe archive member: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError(f"duplicate submission path: {member.filename}")
            with target.open("wb") as output:
                output.write(stream.read(member))


def _merge_submissions(merged_root, spec):
    if spec["name"] != "mosev2" or spec["split"] != "valid":
        return
    combo_name = f"{spec['name']}_{spec['split']}"
    for combo in sorted(merged_root.glob(f"{combo_name}/*")):
        if not combo.is_dir():
            continue
        model = combo.name
        for condition in ("cold", "full_memory"):
            archives = sorted(combo.glob(f"submission-{condition}-*.zip"))
            staging = combo / f".merged-submission-{condition}"
            if staging.exists():
                shutil.rmtree(staging)
            missing = not archives
            if archives:
                staging.mkdir(parents=True)
                try:
                    for archive in archives:
                        _safe_extract(archive, staging)
                except (OSError, ValueError, zipfile.BadZipFile):
                    missing = True
            output_base = combo / f"mosev2-valid-{model}-{condition}"
            output_zip = output_base.with_suffix(".zip")
            output_zip.unlink(missing_ok=True)
            if not missing:
                shutil.make_archive(str(output_base), "zip", staging)
            if staging.exists():
                shutil.rmtree(staging)
            _write_json(combo / f"submission-{condition}.json", {
                "condition": condition,
                "archive": str(output_zip) if not missing else None,
                "complete": not missing,
                "source_archives": [str(path) for path in archives],
            })


def merge_spec_results(results_root, spec):
    """Merge every matching shard run and return its complete summary."""
    results_root = Path(results_root).resolve()
    run_roots = []
    for candidate in sorted(results_root.iterdir() if results_root.exists() else []):
        run_json_path = candidate / "run.json"
        if candidate.name.startswith("merged_") or not run_json_path.is_file():
            continue
        try:
            run_json = json.loads(run_json_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if _matches(run_json, spec):
            run_roots.append((candidate, run_json))
    if not run_roots:
        raise FileNotFoundError(f"no local result runs for {spec['name']}/{spec['split']}")

    merged_id = f"merged_{spec['name']}_{spec['split']}"
    merged_root = results_root / merged_id
    if merged_root.exists():
        shutil.rmtree(merged_root)
    merged_root.mkdir(parents=True)

    first_run = run_roots[0][1]
    config = dict(first_run["config"])
    config["output"] = str(results_root)
    config["datasets"] = [dict(spec)]
    if spec["name"] == "lvos":
        config["datasets"][0]["version"] = "v1"
    models = sorted({
        model.name
        for source, _ in run_roots
        for model in ((source / f"{spec['name']}_{spec['split']}").iterdir()
                      if (source / f"{spec['name']}_{spec['split']}").is_dir() else [])
        if model.is_dir()
    })
    if models:
        config["models"] = models
    merged_run = dict(first_run)
    merged_run["run_id"] = merged_id
    merged_run["config"] = config
    merged_run["merged_from"] = [source.name for source, _ in run_roots]
    _write_json(merged_root / "run.json", merged_run)

    combo_name = f"{spec['name']}_{spec['split']}"
    for source, _ in run_roots:
        source_combo = source / combo_name
        if not source_combo.is_dir():
            continue
        for model_dir in sorted(path for path in source_combo.iterdir() if path.is_dir()):
            destination = merged_root / combo_name / model_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            selected = set()
            selection = model_dir / "selection.json"
            if selection.is_file():
                selected.update(json.loads(selection.read_text()).get("videos", []))
            source_videos = model_dir / "videos"
            if source_videos.is_dir():
                for source_video in sorted(path for path in source_videos.iterdir()
                                           if path.is_dir()):
                    target_video = destination / "videos" / source_video.name
                    target_video.mkdir(parents=True, exist_ok=True)
                    selected.add(source_video.name)
                    for filename in ("result.json", "error.json"):
                        source_file = source_video / filename
                        if source_file.is_file():
                            shutil.copy2(source_file, target_video / filename)
            combo_error = model_dir / "error.json"
            if combo_error.is_file():
                shutil.copy2(combo_error, destination / "error.json")
            for archive in model_dir.glob("submission-*.zip"):
                if archive.name.endswith(".zip"):
                    target = destination / f"submission-{archive.stem.split('submission-', 1)[-1]}-{source.name}.zip"
                    shutil.copy2(archive, target)
            _write_json(destination / "selection.json", {
                "videos": sorted(selected), "partial_run": False,
            })

        for error in (source / "errors").glob("*.json"):
            target = merged_root / "errors" / f"{source.name}-{error.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(error, target)

    _merge_submissions(merged_root, spec)
    config_path = results_root / f".{merged_id}.config.json"
    _write_json(config_path, config)
    summary = summarize_run(config_path, merged_id)
    _write_json(merged_root / "run-status.json", {
        "run_id": merged_id,
        "completed": summary["completed_video_jobs"],
        "failed": summary["failed_video_jobs"],
        "failed_combinations": len(summary["combination_failures"]),
    })
    return summary
