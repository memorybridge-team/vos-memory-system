#!/usr/bin/env python3
"""Fresh training plus handoff evaluation within one wall-time budget (default: five hours)."""
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from core import ROOT, METHODS, read, write, sha

QUICK_METHODS = ("small_only", "base_native", "direct", "affine", "residual_mlp", "transformer")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True, help="video-disjoint training/validation pair selection")
    p.add_argument("--selection", type=Path, required=True, help="test10.v1 evaluation manifest disjoint from both training splits")
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs/budget5h")
    p.add_argument("--budget-hours", type=float, default=5.)
    p.add_argument("--training-hours", type=float, default=3.5, help="maximum training allocation; remaining time goes to evaluation")
    p.add_argument("--epochs", type=int, default=30, help="upper bound; time budget may stop within an epoch")
    p.add_argument("--batch-records", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--validation-pairs", type=int, default=64)
    p.add_argument("--validate-every", type=int, default=2048)
    p.add_argument("--cache-gib", type=float, default=2.)
    p.add_argument("--all-methods", action="store_true", help="use manifest.methods (or all original methods) instead of the six core methods")
    return p


def terminate(process):
    # Include child workers in termination; start_new_session gives this process its own group.
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2.)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2.)
    except ProcessLookupError:
        process.wait(timeout=2.)


def run(args):
    start = time.monotonic()
    if (not math.isfinite(args.budget_hours) or not math.isfinite(args.training_hours) or
            not 0 < args.training_hours < args.budget_hours or args.budget_hours * 3600 < 180.):
        raise ValueError("require 0 < training-hours < budget-hours and at least three minutes total")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("use a new, empty output directory")
    selection = read(args.selection)
    if selection.get("schema") != "test10.v1" or not selection.get("cases"):
        raise ValueError("nonempty test10.v1 evaluation manifest required")
    # Freeze the comparison before training; no accuracy-based case/method selection.
    selection = dict(selection)
    if not args.all_methods:
        selection["methods"] = list(QUICK_METHODS)
    seed = selection["seed"]
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_selection = args.output_dir / "evaluation_selection.json"
    write(frozen_selection, selection)
    training_dir = args.output_dir / "training"
    eval_dir = args.output_dir / "evaluation"
    deadline = start + args.budget_hours * 3600
    # Reserve a minute for rebuilding the summary, and ten seconds for final process cleanup.
    work_deadline = deadline - 70.
    pipeline = dict(schema="test10.budget_pipeline.v1", status="running", budget_hours=args.budget_hours,
                    training_hours=args.training_hours, report_reserve_seconds=70,
                    source=dict(selection=str(args.selection.resolve()), selection_sha256=sha(args.selection),
                                pairs=str(args.pairs.resolve()), pairs_sha256=sha(args.pairs),
                                pipeline_sha256=sha(Path(__file__))),
                    methods=selection.get("methods", list(METHODS)), stages=[])
    report_path = args.output_dir / "pipeline_report.json"
    write(report_path, pipeline)

    def execute(name, command, end):
        remaining = end - time.monotonic()
        if remaining <= 0:
            pipeline["stages"].append(dict(stage=name, status="not_started", reason="time_budget"))
            write(report_path, pipeline)
            return False
        stage_start = time.monotonic()
        row = dict(stage=name, status="running", command=command, timeout_seconds=remaining)
        pipeline["stages"].append(row); write(report_path, pipeline)
        print(f"[{name}] remaining allocation: {remaining / 60:.1f} minutes", flush=True)
        with (args.output_dir / f"{name}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                row["returncode"] = process.wait(timeout=max(.01, end - time.monotonic()))
                row["status"] = "finished" if row["returncode"] == 0 else "failed"
            except subprocess.TimeoutExpired:
                terminate(process)
                row.update(status="time_budget", returncode=process.returncode)
            except BaseException:
                terminate(process)
                row["status"] = "interrupted"
                raise
            finally:
                row["seconds"] = time.monotonic() - stage_start
                write(report_path, pipeline)
        return row["status"] == "finished"

    try:
        train_end = min(start + args.training_hours * 3600, work_deadline)
        # Account for interpreter/torch startup in the parent's limit; leave the child exit margin.
        train_seconds = train_end - time.monotonic() - 15.
        command = [sys.executable, str(ROOT / "training.py"), "--pairs", str(args.pairs.resolve()),
                   "--output-dir", str(training_dir), "--device", "cuda", "--seed", str(seed),
                   "--budget-hours", str(max(.001, train_seconds) / 3600), "--epochs", str(args.epochs),
                   "--batch-records", str(args.batch_records), "--learning-rate", str(args.learning_rate),
                   "--validation-pairs", str(args.validation_pairs), "--validate-every", str(args.validate_every),
                   "--cache-gib", str(args.cache_gib)]
        trained = execute("training", command, train_end)
        if trained:
            # One frozen inference budget for smoke/full. Audit and process startup are also
            # charged by the outer deadline, irrespective of run.py's inference-only counter.
            eval_hours = max(.001, work_deadline - time.monotonic()) / 3600
            common = [sys.executable, str(ROOT / "run.py"), "--run-dir", str(eval_dir),
                      "--training-dir", str(training_dir), "--selection", str(frozen_selection),
                      "--seed", str(seed), "--budget-hours", str(eval_hours)]
            for name in ("audit", "smoke", "full"):
                cmd = [*common, "--stage", name, *([] if name == "audit" else ["--resume"])]
                if not execute(name, cmd, work_deadline):
                    break
        pipeline["status"] = "finished" if pipeline["stages"][-1]["stage"] == "full" and pipeline["stages"][-1]["status"] == "finished" else "incomplete"
    finally:
        # Even a killed inference case leaves atomic rows for completed cases. Rebuild only
        # from cases where all selected methods AND the injection gate have finished.
        if (eval_dir / "provenance.json").exists() and time.monotonic() < deadline - 10.:
            execute("summary", [sys.executable, str(ROOT / "run.py"), "--stage", "full",
                               "--report-only", "--run-dir", str(eval_dir), "--seed", str(seed)], deadline - 10.)
        summary_path = eval_dir / "summary.json"
        if summary_path.exists():
            summary = read(summary_path)
            pipeline["evaluation"] = dict(candidate_cases=summary["candidate_cases"],
                complete_cases=summary["complete_cases"], incomplete_cases=len(summary["incomplete"]),
                all_candidates_completed=summary["complete_cases"] == summary["candidate_cases"])
        train_report = training_dir / "train_report.json"
        if train_report.exists():
            fit = read(train_report)
            pipeline["training"] = dict(status=fit["status"], attempts=fit.get("attempts", {}),
                                         integrity=fit.get("integrity"))
        pipeline["seconds"] = time.monotonic() - start
        pipeline["within_budget"] = pipeline["seconds"] <= args.budget_hours * 3600
        if pipeline["status"] == "running":
            pipeline["status"] = "interrupted"
        write(report_path, pipeline)
        print(f"Pipeline {pipeline['status']}: {pipeline['seconds'] / 3600:.3f} hours. {report_path}", flush=True)
    return pipeline


if __name__ == "__main__":
    result = run(parser().parse_args())
    if result["status"] != "finished":
        raise SystemExit(1)
