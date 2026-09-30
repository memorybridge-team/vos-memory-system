#!/usr/bin/env python3
"""Explicit audit/smoke/full entrypoint; audit never imports GPU runtime."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time
import traceback

from core import (ROOT, WORKSPACE, METHODS, GATE, Artifacts, read, write, digest,
                  select_smoke, freeze_schedule, verify_snapshot)
from manifest import build, provenance, audit_legacy, imports
from report import build_report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("audit", "smoke", "full"), required=True)
    p.add_argument("--selection", type=Path, help="existing test10 manifest; omitted: deterministic candidates")
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs/default")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--budget-hours", type=float, default=4.)
    p.add_argument("--reuse-run", action="append", type=Path, default=[])
    p.add_argument("--report-only", action="store_true", help="rebuild reports without loading any GPU model")
    return p


def verify_fit_store():
    imports()
    from vos_memory_inspector.paired_state_store import PairedStateStore
    store = PairedStateStore(WORKSPACE / "test9/mvp_store", create=False)
    shards = list(store.index(split="fit", verify_checksum=True))
    return dict(shards=len(shards), records=sum(s["records"] for s in shards),
                videos={d: sorted({s["video_id"] for s in shards if s["dataset"] == d}) for d in ("MOSEv2", "LVOSv2")},
                fingerprint=store.fingerprint(split="fit"))


def run(args):
    if args.budget_hours <= 0: raise ValueError("budget must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    # Concurrent execution would corrupt the wall-time budget; hold a single-writer lock.
    with (args.run_dir / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return locked_run(args)


def locked_run(args):
    meta_path = args.run_dir / "provenance.json"
    manifest_path = args.run_dir / "selection.json"
    if meta_path.exists():
        if not args.resume and not args.report_only:
            raise ValueError("existing run: use --resume or a new --run-dir")
        prov = read(meta_path); verify_snapshot(prov["files"])
        selection = read(manifest_path)
        if args.selection and digest(read(args.selection)) != digest(selection):
            raise ValueError("selection differs from the saved run")
    else:
        if args.stage != "audit": raise ValueError("run --stage audit first")
        prov = provenance()
        selection = read(args.selection) if args.selection else build(args.seed)
        if selection.get("schema") != "test10.v1": raise ValueError("expected a test10 manifest")
        # Selection is committed before provenance; incomplete setup can be retried safely.
        write(manifest_path, selection)
        write(meta_path, prov)
    if args.seed != selection["seed"]: raise ValueError("seed differs from frozen manifest")
    store = Artifacts(args.run_dir, prov, args.reuse_run)
    if args.report_only:
        return build_report(selection, store, args.seed)
    if args.stage == "audit":
        fit = verify_fit_store()
        from core import LEGACY
        trained = read(LEGACY / "train/train_report.json")
        if fit["fingerprint"] != trained["store_fingerprint"]:
            raise ValueError("fit store differs from trained checkpoint provenance")
        for dataset, videos in fit["videos"].items():
            eval_videos = {c["video_id"] for c in selection["cases"] if c["dataset"] == dataset}
            if set(videos) & eval_videos:
                raise ValueError(f"fit/evaluation video overlap: {dataset}")
        legacy = audit_legacy(selection, prov)
        write(args.run_dir / "legacy_audit.json", legacy)
        smoke = select_smoke(selection["cases"], args.seed)
        write(args.run_dir / "smoke_selection.json", {"case_ids": [c["case_id"] for c in smoke]})
        write(args.run_dir / "audit.json", dict(fit=fit,
            candidate_counts={d: sum(c["dataset"]==d for c in selection["cases"]) for d in ("MOSEv2","LVOSv2","DAVIS2017")},
            missing=[{"case_id": c["case_id"], "methods": [m for m in (*METHODS,GATE) if store.load(c,m) is None]}
                     for c in selection["cases"]],
            legacy_unverified=sum(r["status"]=="unverified" for r in legacy),
            note="Legacy rows retained for historical comparison; missing past input hashes prevent verified score reuse. No training or inference performed."))
        print(json.dumps(read(args.run_dir / "audit.json")["candidate_counts"]), flush=True)
        return
    if not (args.run_dir / "audit.json").exists(): raise ValueError("completed audit required")
    if args.stage == "full" and not (args.run_dir / "smoke_gate.json").exists():
        raise ValueError("32-case smoke gate must pass before full")
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
        evaluator = Evaluator(args.seed)
        if args.stage == "smoke":
            queue = [{"case_id": cid} for cid in smoke_ids]
        else:
            schedule_path = args.run_dir / "schedule.json"
            if schedule_path.exists():
                schedule = read(schedule_path)
            else:
                candidates = [c for c in selection["cases"] if c["case_id"] not in smoke_ids
                              and not all(store.load(c,m) for m in (*METHODS,GATE))]
                # Reserve half an hour for completion/retry; never select using scores.
                remaining = total_limit - before - (time.monotonic() - session_start) - 1800
                schedule = freeze_schedule(candidates, timings, remaining, args.seed)
                write(schedule_path, schedule)
            queue = schedule
        for item in queue:
            case = by_id[item["case_id"]]
            complete = all(store.load(case,m) for m in (*METHODS,GATE))
            if complete and (args.stage != "smoke" or store.load(case,"costs")): continue
            elapsed = before + time.monotonic() - session_start
            if elapsed >= total_limit: break
            if args.stage == "full" and elapsed >= total_limit - 1800 and item["estimated_seconds"] * 1.2 > total_limit - elapsed:
                break
            t = time.monotonic()
            try:
                verify_snapshot(prov["files"])
                result = evaluator.case(case, store, measure=args.stage=="smoke")
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
            passed = all(all(store.load(by_id[cid],m) for m in (*METHODS,GATE,"costs")) for cid in smoke_ids)
            if passed:
                write(args.run_dir / "smoke_gate.json", {"passed": True, "cases": smoke_ids})
    finally:
        budget["elapsed_seconds"] = before + time.monotonic() - session_start
        budget.pop("active_since", None); write(clock_path, budget)
        build_report(selection, store, args.seed, scheduled={x["case_id"] for x in schedule} if schedule else set(smoke_ids))


if __name__ == "__main__":
    run(parser().parse_args())
