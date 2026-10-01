#!/usr/bin/env python3
"""Explicit audit/smoke/full entrypoint; audit never imports GPU runtime."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time
import traceback
from statistics import median

from core import (ROOT, WORKSPACE, TRAINING, METHODS, GATE, SWITCH_WINDOW, Artifacts, read, write, digest,
                  select_smoke, freeze_schedule, verify_snapshot, harness_excluded, snapshot)
from manifest import build, provenance, audit_legacy, imports
from report import build_report


def heldout_smoke(cases):
    return [c for c in cases if c.get("cohort") == "heldout_core" and c.get("slot") in (0, 1, 2)]


def heldout_schedule(cases, timings, remaining_seconds):
    """Freeze equal-dataset core cases first; use time estimates only for extension."""
    def estimate(case):
        same = [r["seconds"] for r in timings if r["dataset"] == case["dataset"]
                and r["length_bin"] == case["length_bin"]]
        if not same:
            same = [r["seconds"] for r in timings if r["dataset"] == case["dataset"]]
        if not same: raise ValueError(f"missing smoke timing: {case['dataset']}")
        return median(same)
    ordered = sorted(cases, key=lambda c: (c["slot"], ("MOSEv2", "LVOSv2", "DAVIS2017", "VOST").index(c["dataset"])))
    core = [dict(case_id=c["case_id"], estimated_seconds=estimate(c)) for c in ordered
            if c["cohort"] == "heldout_core"]
    extension = [dict(case_id=c["case_id"], estimated_seconds=estimate(c)) for c in ordered
                 if c["cohort"] == "heldout_extension"]
    # Optional work starts only when all 20 predeclared cases fit with 20% margin.
    if sum(x["estimated_seconds"] for x in (*core, *extension)) * 1.2 <= remaining_seconds - 900:
        return core + extension
    return core


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("audit", "smoke", "full", "rescore"), required=True,
                   help="rescore: CPU-only rescoring of saved predictions (adds the switch-window metric)")
    p.add_argument("--selection", type=Path, help="existing test10 manifest; omitted: deterministic candidates")
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs/default")
    p.add_argument("--training-dir", type=Path,
                   help="completed fresh prepared-state-pair training; no legacy fallback")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--budget-hours", type=float, default=4.)
    p.add_argument("--reuse-run", action="append", type=Path, default=[])
    p.add_argument("--report-only", action="store_true", help="rebuild reports without loading any GPU model")
    return p


def verify_training_collection(directory):
    from training import training_report
    report = training_report(directory)
    rows = report["collection"]["pairs"]
    # Validation also influenced checkpoint selection, so exclude both splits from evaluation.
    videos = {}
    for row in rows:
        videos.setdefault(row["dataset"], set()).add(row["video_id"])
    return dict(pairs=len(rows), records=sum(r["valid_records"] for r in rows),
                videos={d: sorted(v) for d, v in videos.items()},
                fingerprint=report["collection"]["fingerprint"])


def run(args):
    if args.budget_hours <= 0: raise ValueError("budget must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    # Concurrent execution would corrupt the wall-time budget; hold a single-writer lock.
    with (args.run_dir / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return locked_run(args)


def rescore(selection, store, methods):
    """Rescore saved rows from cached predictions without loading any model."""
    from runtime import rescore_case
    started = time.monotonic()
    rows = sum(rescore_case(c, store, methods) for c in selection["cases"]
               if any(store.load(c, m) for m in methods))
    write(store.root / "rescore.json", dict(switch_window=SWITCH_WINDOW, rows=rows,
                                            seconds=time.monotonic() - started,
                                            harness=snapshot(ROOT.glob("*.py")),
                                            note="scores recomputed from sha-verified prediction caches; "
                                                 "whole-suffix scores reproduced exactly; no inference"))
    return rows


def locked_run(args):
    meta_path = args.run_dir / "provenance.json"
    manifest_path = args.run_dir / "selection.json"
    offline = args.report_only or args.stage == "rescore"
    if meta_path.exists():
        if not args.resume and not offline:
            raise ValueError("existing run: use --resume or a new --run-dir")
        # Offline stages may run newer harness code; everything else stays pinned.
        prov = read(meta_path); verify_snapshot(harness_excluded(prov["files"]) if offline else prov["files"])
        selection = read(manifest_path)
        if args.selection and digest(read(args.selection)) != digest(selection):
            raise ValueError("selection differs from the saved run")
    else:
        if args.stage != "audit": raise ValueError("run --stage audit first")
        prov = provenance(args.training_dir or TRAINING)
        selection = read(args.selection) if args.selection else build(args.seed)
        if selection.get("schema") != "test10.v1": raise ValueError("expected a test10 manifest")
        # Selection is committed before provenance; incomplete setup can be retried safely.
        write(manifest_path, selection)
        write(meta_path, prov)
    if args.seed != selection["seed"]: raise ValueError("seed differs from frozen manifest")
    if args.training_dir and str(args.training_dir.resolve()) != prov.get("training_dir"):
        raise ValueError("training directory differs from the frozen run; use a new run-dir")
    verify_snapshot(selection.get("source_hashes", {}))
    methods = tuple(selection.get("methods", METHODS))
    if not methods or any(m not in METHODS for m in methods):
        raise ValueError("invalid selected methods")
    if selection.get("design") == "heldout.v1" and not {"small_only", "base_native"}.issubset(methods):
        raise ValueError("heldout requires both native references")
    store = Artifacts(args.run_dir, prov, args.reuse_run)
    if args.stage == "rescore":
        print(json.dumps({"rescored_rows": rescore(selection, store, methods)}), flush=True)
        return build_report(selection, store, args.seed)
    if args.report_only:
        return build_report(selection, store, args.seed)
    if args.stage == "audit":
        fit = verify_training_collection(prov.get("training_dir", TRAINING))
        for dataset, videos in fit["videos"].items():
            eval_videos = {c["video_id"] for c in selection["cases"] if c["dataset"] == dataset}
            if set(videos) & eval_videos:
                raise ValueError(f"training/validation and evaluation video overlap: {dataset}")
        if any(c["end"] - c["switch"] < SWITCH_WINDOW for c in selection["cases"]):
            raise ValueError("new evaluation requires all ten post-switch frames in every case")
        for method in methods:
            if method.startswith("anchor_replay_"):
                count = int(method.rsplit("_", 1)[1])
                if any(c["switch"] - c["first"] < count for c in selection["cases"]):
                    raise ValueError(f"insufficient prefix frames for {method}; remove it from selection.methods")
        for case in selection["cases"]:
            if case.get("state_pair_path"):
                from state_pairs import load_pair, verify_case_pair
                if not case.get("state_pair_sha256"):
                    raise ValueError("external state pairs require state_pair_sha256")
                source, _, metadata = load_pair(case["state_pair_path"], expected_sha=case["state_pair_sha256"])
                verify_case_pair(case, source, metadata)
        legacy = audit_legacy(selection, prov)
        write(args.run_dir / "legacy_audit.json", legacy)
        smoke = heldout_smoke(selection["cases"]) if selection.get("design") == "heldout.v1" else select_smoke(selection["cases"], args.seed)
        write(args.run_dir / "smoke_selection.json", {"case_ids": [c["case_id"] for c in smoke]})
        write(args.run_dir / "audit.json", dict(fit=fit,
            candidate_counts={d: sum(c["dataset"]==d for c in selection["cases"]) for d in dict.fromkeys(c["dataset"] for c in selection["cases"])},
            missing=[{"case_id": c["case_id"], "methods": [m for m in (*methods,GATE) if store.load(c,m) is None]}
                     for c in selection["cases"]],
            legacy_unverified=sum(r["status"]=="unverified" for r in legacy),
            note="Legacy rows retained for historical comparison; missing past input hashes prevent verified score reuse. No training or inference performed."))
        print(json.dumps(read(args.run_dir / "audit.json")["candidate_counts"]), flush=True)
        return
    if not (args.run_dir / "audit.json").exists(): raise ValueError("completed audit required")
    if args.stage == "full" and not (args.run_dir / "smoke_gate.json").exists():
        raise ValueError("smoke gate must pass before full")
    clock_path = args.run_dir / "budget.json"
    budget = read(clock_path) if clock_path.exists() else {"elapsed_seconds": 0., "hours": args.budget_hours, "attempts": []}
    if budget["hours"] != args.budget_hours: raise ValueError("budget is frozen; use a new run to change it")
    total_limit = args.budget_hours * 3600
    if budget["elapsed_seconds"] >= total_limit: raise ValueError("inference budget exhausted")
    # Running/interrupt marker prevents a killed process from silently losing charged time.
    if budget.get("active_since"):
        budget["elapsed_seconds"] += max(0., time.time() - budget.pop("active_since"))
        write(clock_path, budget)
        if budget["elapsed_seconds"] >= total_limit:
            raise ValueError("budget exhausted after interrupted session (downtime conservatively charged)")
    before = budget["elapsed_seconds"]
    budget["active_since"] = time.time(); write(clock_path, budget)
    session_start = time.monotonic()
    timings_path = args.run_dir / "case_timings.json"
    timings = read(timings_path) if timings_path.exists() else []
    smoke_ids = read(args.run_dir / "smoke_selection.json")["case_ids"]
    by_id = {c["case_id"]: c for c in selection["cases"]}
    schedule = None
    try:
        from runtime import Evaluator
        import torch
        device = dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                      torch=torch.__version__, cuda=torch.version.cuda)
        device_path = args.run_dir / "device.json"
        if device_path.exists() and read(device_path) != device:
            raise ValueError("GPU/software environment changed; use a new run")
        write(device_path, device)
        evaluator = Evaluator(args.seed, training_dir=Path(prov["training_dir"]))
        if args.stage == "smoke":
            queue = [{"case_id": cid} for cid in smoke_ids]
        else:
            schedule_path = args.run_dir / "schedule.json"
            if schedule_path.exists():
                schedule = read(schedule_path)
            else:
                candidates = [c for c in selection["cases"] if c["case_id"] not in smoke_ids
                              and not all(store.load(c,m) for m in (*methods,GATE))]
                # Reserve half an hour for completion/retry; never select using scores.
                remaining = total_limit - before - (time.monotonic() - session_start) - 1800
                schedule = (heldout_schedule(candidates, timings, total_limit - before - (time.monotonic() - session_start))
                            if selection.get("design") == "heldout.v1" else
                            freeze_schedule(candidates, timings, remaining, args.seed))
                write(schedule_path, schedule)
            queue = schedule
        for item in queue:
            case = by_id[item["case_id"]]
            complete = all(store.load(case,m) for m in (*methods,GATE))
            if complete and (args.stage != "smoke" or store.load(case,"costs")): continue
            elapsed = before + time.monotonic() - session_start
            if elapsed >= total_limit: break
            if args.stage == "full" and elapsed >= total_limit - 1800 and item["estimated_seconds"] * 1.2 > total_limit - elapsed:
                break
            t = time.monotonic()
            try:
                verify_snapshot(prov["files"])
                result = evaluator.case(case, store, measure=args.stage=="smoke", methods=methods)
                timings.append(result); write(timings_path, timings)
                budget["attempts"].append(dict(case_id=case["case_id"], status="complete", seconds=time.monotonic()-t))
                print(json.dumps(result), flush=True)
            except Exception as exc:
                budget["attempts"].append(dict(case_id=case["case_id"], status="failed", seconds=time.monotonic()-t,
                                                error=repr(exc), traceback=traceback.format_exc()))
                # Gate or model errors stop the queue; a resume retries only missing work.
                raise
            finally:
                budget["elapsed_seconds"] = before + time.monotonic() - session_start
                budget["active_since"] = time.time(); write(clock_path, budget)
        if args.stage == "smoke":
            passed = all(all(store.load(by_id[cid],m) for m in (*methods,GATE,"costs")) for cid in smoke_ids)
            if passed:
                write(args.run_dir / "smoke_gate.json", {"passed": True, "cases": smoke_ids})
    finally:
        budget["elapsed_seconds"] = before + time.monotonic() - session_start
        budget.pop("active_since", None); write(clock_path, budget)
        build_report(selection, store, args.seed, scheduled={x["case_id"] for x in schedule} if schedule else set(smoke_ids))


if __name__ == "__main__":
    run(parser().parse_args())
