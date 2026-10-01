"""Pure-Python contracts, provenance, atomic artifacts and budget planning."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("TEST10_WORKSPACE", ROOT.parent)).resolve()
REPO = Path(os.environ.get("TEST10_TRANSLATOR_REPO", WORKSPACE / "test9/vos-memory-translator-nonlinear")).resolve()
if not REPO.is_dir() and "TEST10_TRANSLATOR_REPO" not in os.environ:
    REPO = ROOT.parent.parent / "vos-memory-translator-nonlinear"
TRAINING = ROOT / "training/default"
LEGACY = WORKSPACE / "test9/mvp_runs/2026-09-28_mvp"
METHODS = ("small_only", "base_native", "direct", "affine", "affine_spatial",
           "affine_pointer", "residual_mlp", "transformer", "last_mask",
           "anchor_replay_4", "anchor_replay_8", "anchor_replay_16")
GATE = "self_injection"
DATASETS = ("MOSEv2", "LVOSv2", "DAVIS2017")
WEIGHTS = {"MOSEv2": .4, "LVOSv2": .4, "DAVIS2017": .2}
PROTOCOL = "test10.v1"
# Evaluate each of the first ten processed frames, including missing-GT status.
# This is an evaluation horizon, not the SAM 2 memory lifetime (num_maskmem=7).
SWITCH_WINDOW = 10
SCORE_ROW_FIELDS = ("case_id", "dataset", "video_id", "object_id", "cohort", "checkpoint_video",
                    "method", "key", "status", "scores")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def rank(seed, *parts):
    return digest([seed, *parts])


def positions(stems, prompt, switch, end):
    lookup = {int(s): i for i, s in enumerate(stems)}
    first, old, last = (lookup[int(s)] for s in (prompt, switch, end))
    new = max(old, first + 16)
    if new >= last:
        raise ValueError("no suffix after anchor + 16 frames")
    return first, new, last


def replay_frames(first, switch, k):
    if k not in (4, 8, 16) or switch - first < k:
        raise ValueError("anchor must precede the entire recent replay window")
    return [first, *range(switch - k + 1, switch + 1)]


def switch_window(case, k=SWITCH_WINDOW):
    """Processed-frame positions scored as 'at the switch': the first k Base+ outputs.

    The switch frame itself is the last Source (Small) output and is never scored.
    LVOS/VOST offsets are processed frames (raw stride 5/6)."""
    if k < 1:
        raise ValueError("switch window must contain at least one frame")
    return list(range(case["switch"] + 1, min(case["switch"] + k, case["end"]) + 1))


def suffix_scores(case, masks, score_positions, k=SWITCH_WINDOW):
    """Whole-suffix scores plus the switch-window score, both via the same scorer."""
    if set(masks) != set(range(case["switch"] + 1, case["end"] + 1)):
        raise ValueError("suffix frame mismatch")
    scores = score_positions(masks)
    window = switch_window(case, k)
    early = score_positions({p: masks[p] for p in window})["post_switch"]
    scores["switch_window"] = dict(
        definition=f"annotated processed frames switch+1..switch+{k} (first {k} Base+ outputs)",
        k=k, offsets=[p - case["switch"] for p in window],
        frames=early["frames"], J=early["J"], F=early["F"], J_and_F=early["J_and_F"])
    per_frame = {}
    for offset in range(1, k + 1):
        position = case["switch"] + offset
        row = dict(offset=offset, position=position, frame_stem=None,
                   status="outside_suffix", frames=0, J=None, F=None, J_and_F=None)
        if position <= case["end"]:
            metric = score_positions({position: masks[position]})["post_switch"]
            row.update(frame_stem=case["frame_stems"][position], **metric)
            row["status"] = "scored" if metric["frames"] else "missing_annotation"
        per_frame[f"+{offset}"] = row
    scores["switch_frames"] = per_frame
    return scores


def _same(a, b, tol=1e-9):
    return a == b if a is None or b is None else abs(a - b) <= tol


def rescore_case(case, store, methods, masks_for, score_masks):
    """Recompute saved score rows from cached predictions; never runs a model.

    The whole-suffix score must reproduce exactly, otherwise the cache or scorer drifted."""
    updated = 0
    for method in methods:
        row = store.load(case, method)
        if row is None:
            continue
        scores = score_masks(case, method, masks_for(method))
        old, new = row["scores"]["post_switch"], scores["post_switch"]
        if old["frames"] != new["frames"] or not all(_same(old[f], new[f]) for f in ("J", "F", "J_and_F")):
            raise ValueError(f"suffix score changed on rescore: {case['case_id']} {method}")
        kept = {k: v for k, v in row.items() if k not in SCORE_ROW_FIELDS}
        store.save(case, method, **kept, scores=scores)
        updated += 1
    return updated


def case_contract(case):
    # Selection order, cohort and filenames do not affect inference semantics.
    keys = ("dataset", "video_id", "object_id", "first", "switch", "end", "frame_stems",
            "input_sha256", "annotation_sha256", "sampling")
    contract = {k: case[k] for k in keys}
    if "state_pair_sha256" in case:
        contract["state_pair_sha256"] = case["state_pair_sha256"]
    return contract


def key(case, method, provenance):
    return digest({"protocol": PROTOCOL, "case": case_contract(case), "method": method,
                   "provenance": provenance})


class Artifacts:
    def __init__(self, root, provenance, reuse_roots=()):
        self.root, self.provenance = Path(root), provenance
        self.reuse_roots = tuple(map(Path, reuse_roots))

    def path(self, case, method, suffix=".json"):
        return self.root / "artifacts" / (key(case, method, self.provenance) + suffix)

    def load(self, case, method):
        path = self.path(case, method)
        if not path.exists():
            for root in self.reuse_roots:
                previous = root / "artifacts" / path.name
                if previous.is_file():
                    row = read(previous)
                    if row.get("key") == key(case, method, self.provenance) and row.get("status") == "complete":
                        row = dict(row, origin="reused", reused_from=str(previous),
                                   case_id=case["case_id"], cohort=case["cohort"],
                                   checkpoint_video=case["checkpoint_video"])
                        write(path, row)
                        return row
            return None
        row = read(path)
        if row.get("key") != key(case, method, self.provenance):
            raise ValueError(f"artifact contract mismatch: {path}")
        if row.get("status") != "complete":
            return None
        return dict(row, case_id=case["case_id"], cohort=case["cohort"],
                    checkpoint_video=case["checkpoint_video"])

    def save(self, case, method, **data):
        row = dict(case_id=case["case_id"], dataset=case["dataset"], video_id=case["video_id"],
                   object_id=case["object_id"], cohort=case["cohort"],
                   checkpoint_video=case["checkpoint_video"], method=method,
                   key=key(case, method, self.provenance), status="complete", **data)
        write(self.path(case, method), row)
        return row


def select_smoke(cases, seed=7):
    selected = []
    # Eight changed legacy cases where available; LVOS may have none.
    changed = sorted((c for c in cases if c.get("changed")), key=lambda c: rank(seed, c["case_id"]))
    quotas = {"MOSEv2": 12, "LVOSv2": 12, "DAVIS2017": 8}
    for c in changed:
        if len(selected) == 8:
            break
        if quotas[c["dataset"]] > 0:
            selected.append(c); quotas[c["dataset"]] -= 1
    for dataset, count in quotas.items():
        picked = {c["case_id"] for c in selected}
        pool = sorted((c for c in cases if c["dataset"] == dataset and c["case_id"] not in picked),
                      key=lambda c: (c["cohort"] != "additional", rank(seed, c["case_id"])))
        # Cycle length bins instead of preferring short videos.
        bins = [[c for c in pool if c["length_bin"] == b] for b in range(3)]
        while count and any(bins):
            for bucket in bins:
                if count and bucket:
                    selected.append(bucket.pop(0)); count -= 1
        if count:
            raise ValueError(f"insufficient smoke candidates in {dataset}: missing {count}")
    return selected


def freeze_schedule(cases, timings, remaining_seconds, seed=7):
    """Predeclare a stratified queue; never consume accuracy scores for selection."""
    if not timings:
        raise ValueError("smoke timing is required")
    def estimate(c):
        same = [t["seconds"] for t in timings if t["dataset"] == c["dataset"] and t["length_bin"] == c["length_bin"]]
        if not same:
            same = [t["seconds"] for t in timings if t["dataset"] == c["dataset"]]
        return max(1., median(same or [t["seconds"] for t in timings]))
    queues = {}
    for d in DATASETS:
        pool = sorted((c for c in cases if c["dataset"] == d),
                      key=lambda c: (c["cohort"] != "additional", rank(seed, c["case_id"])))
        # Deterministic length-stratified ordering, new videos first.
        q = []
        for additional in (True, False):
            bins = [[c for c in pool if (c["cohort"] == "additional") == additional and c["length_bin"] == b] for b in range(3)]
            while any(bins):
                for bucket in bins:
                    if bucket: q.append(bucket.pop(0))
        queues[d] = q
    spent = {d: 0. for d in DATASETS}
    budget, schedule = max(0., remaining_seconds) * .8, []
    while any(queues.values()):
        d = min((d for d in DATASETS if queues[d]), key=lambda d: spent[d] / WEIGHTS[d])
        c = queues[d].pop(0); cost = estimate(c)
        if cost > budget:
            # Do not scan this stratum for a cheaper/shorter replacement.
            queues[d] = []
            continue
        schedule.append({"case_id": c["case_id"], "estimated_seconds": cost})
        budget -= cost; spent[d] += cost
    return schedule


def snapshot(paths):
    return {str(Path(p).resolve()): sha(p) for p in sorted(set(map(Path, paths)))}


def harness_excluded(files):
    """Offline report/rescore may use newer test10 harness code; models, checkpoints,
    SAM 2 and test9 scoring code remain pinned by the remaining hashes."""
    root = ROOT.resolve()
    return {p: h for p, h in files.items() if not (Path(p).parent == root and Path(p).suffix == ".py")}


def verify_snapshot(files):
    changed = [p for p, expected in files.items() if not Path(p).is_file() or sha(p) != expected]
    if changed:
        raise ValueError(f"referenced inputs changed: {changed[:5]}")
