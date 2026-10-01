#!/usr/bin/env python3
"""At a completed core boundary, admit all predeclared extensions using costs only.

Must run with the evaluator stopped. Does not add, replace or reselect videos.
Accuracy is never consulted. A safety reserve and 20% estimate margin apply.
"""
import argparse
import fcntl
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import Artifacts, GATE, digest, read, verify_snapshot, write
from run import restore_selection_paths

DATASETS = ("MOSEv2", "LVOSv2", "DAVIS2017", "VOST")


def extension_plan(cases, timings, available, reserve=1800):
    extension = sorted((c for c in cases if c["cohort"] == "heldout_extension"),
                       key=lambda c: (c["slot"], DATASETS.index(c["dataset"])))
    assert len(extension) == 20
    assert all(sum(c["dataset"] == d for c in extension) == 5 for d in DATASETS)
    plan = []
    for c in extension:
        costs = [t["seconds"] for t in timings if (t["dataset"], t["length_bin"]) == (c["dataset"], c["length_bin"])]
        if not costs:
            costs = [t["seconds"] for t in timings if t["dataset"] == c["dataset"]]
        assert costs, f"no measured cost for {c['dataset']}"
        plan.append(dict(case_id=c["case_id"], estimated_seconds=median(costs)))
    estimate = sum(r["estimated_seconds"] for r in plan)
    return plan, dict(admitted=estimate * 1.2 + reserve <= available,
                      available_seconds=available, estimated_seconds=estimate,
                      margin=1.2, reserve_seconds=reserve,
                      rule="all 20 predeclared videos or none; dataset/length-bin median runtime only; no accuracy selection")


def extend(run_dir):
    with (run_dir / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = read(run_dir / "budget.json")
        assert "active_since" not in budget, "interrupted or active evaluation must be resolved first"
        selection = restore_selection_paths(read(run_dir / "selection.json"))
        provenance = read(run_dir / "provenance.json")
        verify_snapshot(provenance["files"])
        store = Artifacts(run_dir, provenance)
        for case in selection["cases"]:
            if case["cohort"] != "heldout_core":
                continue
            assert all(store.load(case, m) for m in (*selection["methods"], GATE)), case["case_id"]
            assert store.load(case, GATE)["gate_passed"], case["case_id"]
        old = read(run_dir / "schedule.json")
        extension_ids = {c["case_id"] for c in selection["cases"] if c["cohort"] == "heldout_extension"}
        if extension_ids & {r["case_id"] for r in old}:
            assert extension_ids <= {r["case_id"] for r in old}
            return dict(already_admitted=True)
        plan, decision = extension_plan(selection["cases"], read(run_dir / "case_timings.json"),
                                        budget["hours"] * 3600 - budget["elapsed_seconds"])
        decision.update(original_schedule_digest=digest(old), selection_digest=digest(read(run_dir / "selection.json")),
                        cases=plan, core_complete_before_extension=True)
        write(run_dir / "extension_decision.json", decision)
        if decision["admitted"]:
            write(run_dir / "schedule.core.json", old)
            write(run_dir / "schedule.json", old + plan)
        return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(extend(args.run_dir.resolve()))
