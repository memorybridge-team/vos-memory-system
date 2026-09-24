#!/usr/bin/env python3
"""Run or resume the fixed DAVIS rare-event cached-baseline sweep."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vos_memory_inspector.baseline_sweep import (
    RARE_EVENT_TAGS,
    aggregate_completed,
    cache_checksum_matches,
    case_slug,
    load_complete_suite,
    select_cases,
    selection_manifest,
    write_aggregate_reports,
    write_json_atomic,
)
from vos_memory_inspector.evaluation_manifest import load_evaluation_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume the fixed DAVIS rare-event baseline sweep."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--evaluation-repo", required=True, type=Path)
    parser.add_argument("--case-cache-root", required=True, type=Path)
    parser.add_argument("--suite-root", required=True, type=Path)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--hot-cache-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_sequence(
    dataset_root: Path, hot_cache_root: Path | None, sequence: str
) -> tuple[Path, Path]:
    video = dataset_root / "JPEGImages" / "480p" / sequence
    annotation = dataset_root / "Annotations" / "480p" / sequence
    if not video.is_dir() or not annotation.is_dir():
        raise FileNotFoundError(f"DAVIS sequence is incomplete: {sequence}")
    if hot_cache_root is None:
        return video, annotation
    staged_video = hot_cache_root / "videos" / sequence
    staged_annotation = hot_cache_root / "annotations" / sequence
    shutil.copytree(video, staged_video, dirs_exist_ok=True, copy_function=shutil.copy2)
    shutil.copytree(
        annotation, staged_annotation, dirs_exist_ok=True, copy_function=shutil.copy2
    )
    return staged_video, staged_annotation


def _run(command: list[str], *, cwd: Path) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _prepare_command(
    args: argparse.Namespace,
    *,
    case: dict[str, Any],
    video: Path,
    annotation: Path,
    cache: Path,
) -> list[str]:
    executable = str(Path(sys.executable).with_name("cmmt-sam2-prepare-case"))
    return [
        executable,
        "--sam2-repo",
        str(args.sam2_repo),
        "--source-config",
        args.source_config,
        "--source-checkpoint",
        str(args.source_checkpoint),
        "--source-model-id",
        args.source_model_id,
        "--target-config",
        args.target_config,
        "--target-checkpoint",
        str(args.target_checkpoint),
        "--target-model-id",
        args.target_model_id,
        "--video-dir",
        str(video),
        "--prompt-mask",
        str(annotation / "00000.png"),
        "--object-id",
        str(case["object_id"]),
        "--switch-frame",
        str(case["switch_frame"]),
        "--output",
        str(cache),
        "--report-json",
        str(cache.with_suffix(".prepare.json")),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]


def _suite_command(
    args: argparse.Namespace,
    *,
    video: Path,
    annotation: Path,
    cache: Path,
    suite: Path,
) -> list[str]:
    return [
        sys.executable,
        str((Path(__file__).resolve().parent / "run_cached_baseline_suite.py")),
        "--case-cache",
        str(cache),
        "--sam2-repo",
        str(args.sam2_repo),
        "--target-config",
        args.target_config,
        "--target-checkpoint",
        str(args.target_checkpoint),
        "--target-model-id",
        args.target_model_id,
        "--video-dir",
        str(video),
        "--prompt-mask",
        str(annotation / "00000.png"),
        "--annotation-dir",
        str(annotation),
        "--evaluation-repo",
        str(args.evaluation_repo),
        "--output-dir",
        str(suite),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]


def main() -> None:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    manifest = load_evaluation_manifest(args.manifest)
    cases = select_cases(
        manifest,
        tags=RARE_EVENT_TAGS,
        case_ids=args.case_id,
    )
    if not cases:
        raise ValueError("rare-event selection is empty")
    selection = selection_manifest(
        manifest, cases, tags=RARE_EVENT_TAGS, case_ids=args.case_id
    )
    plan = {
        "selection": selection,
        "case_slugs": [case_slug(case) for case in cases],
        "resume_policy": "skip checksummed cache and complete seven-method suite",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    suite_root = args.suite_root.resolve()
    run_directory = args.run_directory.resolve()
    case_cache_root = args.case_cache_root.resolve()
    hot_cache_root = (
        None if args.hot_cache_root is None else args.hot_cache_root.resolve()
    )
    suite_root.mkdir(parents=True, exist_ok=True)
    run_directory.mkdir(parents=True, exist_ok=True)
    case_cache_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(selection, run_directory / "selection_manifest.json")
    status: dict[str, Any] = {
        "schema_version": "cmmt.cached_baseline_sweep_status.v1",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "state": "running",
        "planned_cases": len(cases),
        "cases": {},
    }
    status_path = run_directory / "sweep_status.json"
    write_json_atomic(status, status_path)

    for index, case in enumerate(cases, start=1):
        slug = case_slug(case)
        suite = suite_root / slug
        cache = case_cache_root / f"{slug}.pt"
        complete = load_complete_suite(suite, case)
        if complete is not None:
            state = "skipped_complete"
            status["cases"][slug] = {"state": state, "updated_at": _utc_now()}
            status["updated_at"] = _utc_now()
            write_json_atomic(status, status_path)
            write_aggregate_reports(
                aggregate_completed(cases, suite_root), run_directory
            )
            print(f"[{index}/{len(cases)}] {slug}: {state}", flush=True)
            continue

        started = time.perf_counter()
        status["cases"][slug] = {"state": "running", "updated_at": _utc_now()}
        status["updated_at"] = _utc_now()
        write_json_atomic(status, status_path)
        try:
            video, annotation = _stage_sequence(
                args.dataset_root.resolve(), hot_cache_root, str(case["sequence"])
            )
            if not cache_checksum_matches(cache):
                _run(
                    _prepare_command(
                        args,
                        case=case,
                        video=video,
                        annotation=annotation,
                        cache=cache,
                    ),
                    cwd=project_root,
                )
            _run(
                _suite_command(
                    args,
                    video=video,
                    annotation=annotation,
                    cache=cache,
                    suite=suite,
                ),
                cwd=project_root,
            )
            if load_complete_suite(suite, case) is None:
                raise RuntimeError(f"suite completion validation failed: {slug}")
        except Exception as error:
            status["state"] = "failed"
            status["failed_case"] = slug
            status["error"] = f"{type(error).__name__}: {error}"
            status["cases"][slug] = {
                "state": "failed",
                "updated_at": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
            }
            status["updated_at"] = _utc_now()
            write_json_atomic(status, status_path)
            write_aggregate_reports(
                aggregate_completed(cases, suite_root), run_directory
            )
            raise
        status["cases"][slug] = {
            "state": "complete",
            "updated_at": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        status["updated_at"] = _utc_now()
        write_json_atomic(status, status_path)
        write_aggregate_reports(aggregate_completed(cases, suite_root), run_directory)
        print(f"[{index}/{len(cases)}] {slug}: complete", flush=True)

    status["state"] = "complete"
    status["completed_at"] = _utc_now()
    status["updated_at"] = _utc_now()
    write_json_atomic(status, status_path)
    final = aggregate_completed(cases, suite_root)
    write_aggregate_reports(final, run_directory)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
