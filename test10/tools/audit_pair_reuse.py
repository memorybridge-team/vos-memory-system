#!/usr/bin/env python3
"""Read-only cache compatibility audit, including native full-suffix logit hashes.

This diagnostic never overwrites evaluation state, predictions or scores.
Nonmatching downloaded states are not failed evaluation attempts.
"""
import argparse
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import Artifacts, GATE, read, sha, write
from run import ensure_local_training_module, restore_selection_paths
ensure_local_training_module()
from state_pairs import load_pair, verify_case_pair, TENSORS


def audit(run_dir, catalog):
    import torch
    from runtime import Evaluator, SharedVideoFrames, load_blob
    started = time.perf_counter()
    selection = restore_selection_paths(read(run_dir / "selection.json"))
    provenance = read(run_dir / "provenance.json")
    store = Artifacts(run_dir, provenance)
    indexed = {}
    for row in read(catalog)["pairs"]:
        m = re.search(r"_obj(\d+)_switch(\d+)\.pt$", row["path"])
        if m:
            indexed[row["dataset"], row["video_id"], int(m[1]), int(m[2])] = row
    evaluator = None
    candidates, unmatched = [], []
    for case in selection["cases"]:
        row = indexed.get((case["dataset"], case["video_id"], case["object_id"], case["switch"]))
        if row is None:
            unmatched.append(case["case_id"])
            continue
        result = dict(case_id=case["case_id"], source_file=Path(row["path"]).name,
                      sha256=row["sha256"], metadata_compatible=False)
        candidates.append(result)
        small, base, metadata = load_pair(row["path"], expected_sha=row["sha256"])
        try:
            verify_case_pair(case, small, metadata)
            result["metadata_compatible"] = True
        except ValueError as exc:
            result.update(verified_reusable=False, reason=str(exc))
            continue
        local = store.path(case, "state_pair", ".pt")
        if not local.exists():
            result.update(verified_reusable=None, reason="local reference is not yet complete")
            continue
        a, b, _ = load_pair(local)
        result["different_source_tensors"] = [n for n in TENSORS if not torch.equal(getattr(small,n), getattr(a,n))]
        result["different_target_tensors"] = [n for n in TENSORS if not torch.equal(getattr(base,n), getattr(b,n))]
        def distance(x, y):
            if x.spatial_memory.shape != y.spatial_memory.shape or x.object_pointer.shape != y.object_pointer.shape:
                return None
            return {n: float((getattr(x,n).float() - getattr(y,n).float()).square().mean())
                    for n in ("spatial_memory", "object_pointer")}
        result["state_mse_to_local_reference"] = dict(
            downloaded_source_to_local_small=distance(small,a),
            downloaded_source_to_local_base=distance(small,b),
            downloaded_target_to_local_base=distance(base,b),
            downloaded_target_to_local_small=distance(base,a))
        if evaluator is None:
            evaluator = Evaluator(selection["seed"], training_dir=Path(provenance["training_dir"]))
        source = load_blob(store, case, "source_prefix")
        native = load_blob(store, case, "base_prefix")
        source["state"], native["state"] = small, base
        frames = SharedVideoFrames(case["video_dir"], device="cuda")
        try:
            predicted = evaluator.method(case, frames, source, native, GATE)
            result["downloaded_target_full_suffix_logit_hash_equal_to_native"] = predicted["hashes"] == native["logit_hashes"]
            result["verified_reusable"] = (not result["different_source_tensors"] and
                                           result["downloaded_target_full_suffix_logit_hash_equal_to_native"])
            result["reason"] = ("exact current-execution state and continuation match" if result["verified_reusable"]
                                else "not numerically equivalent to this execution; retain locally generated evaluation pair")
        finally:
            frames.release(); torch.cuda.empty_cache()
    return dict(catalog_sha256=sha(catalog), code_sha256=sha(Path(__file__)), candidate_pairs=candidates,
                no_video_object_switch_match=unmatched,
                seconds=time.perf_counter()-started,
                note="Matching IDs alone do not prove prompt/preprocessing/numerical execution compatibility. "
                     "No reused target is accepted when it fails the exact native continuation gate. "
                     "This audit diagnoses differences; it does not establish their cause or invalidate the fit training bank.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.run_dir.resolve(), args.catalog.resolve())
    write(args.output, result)
    print({"candidate_pairs": len(result["candidate_pairs"]),
           "verified_reusable": sum(r.get("verified_reusable") is True for r in result["candidate_pairs"]),
           "seconds": result["seconds"]})
