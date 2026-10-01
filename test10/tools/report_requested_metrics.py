#!/usr/bin/env python3
"""CPU-only, same-window metrics; never substitute bank target states for Warm Target."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("J", "F", "J_and_F")
BANK_METHODS = ("direct", "affine", "residual_mlp", "transformer")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retention(method, warm):
    """Ratio of matched population means, NOT Direct-to-native gap recovery."""
    return None if method is None or warm is None or warm <= 0 else 100 * method / warm


def frame_mean(frames, visible=False):
    selected = [f for f in frames if f.get("annotated", True) and
                f.get("J_and_F") is not None and (not visible or f["gt_present"])]
    return dict(frames=len(selected), **{k: mean(f[k] for f in selected) if selected else None for k in FIELDS})


def complete_groups(rows, methods):
    groups = defaultdict(dict)
    for row in rows:
        if row["method"] not in methods:
            continue
        key = row["case_id"]
        if row["method"] in groups[key]:
            raise ValueError(f"duplicate result: {key}/{row['method']}")
        groups[key][row["method"]] = row
    return {k: g for k, g in groups.items() if set(g) == set(methods)}


def target_counts(method, processed, scored, visible):
    # Source Only evaluates identical offsets, but never uses the target model.
    return dict(target_frames_processed=0 if method == "small_only" else processed,
                target_frames_scored_at5=0 if method == "small_only" else scored,
                target_visible_frames_scored_at5=0 if method == "small_only" else visible)


def normalized(row, frames, processed, full=None, first=0):
    post5, visible5 = frame_mean(frames), frame_mean(frames, visible=True)
    metrics = dict(post5=post5, visible5=visible5,
                   post_available=full["post_switch"] if full else post5,
                   visible_available=full["gt_visible"] if full else visible5)
    return dict(case_id=row.get("case_id", row.get("pair_id")), dataset=row["dataset"],
                video_id=row["video_id"], object_id=row["object_id"], method=row["method"],
                evaluation_role=row.get("evaluation_role", "excluded_from_current_training_and_selection"),
                first_prompt_position=first, frames=frames, metrics=metrics,
                evaluation_frames_at5=post5["frames"], visible_evaluation_frames_at5=visible5["frames"],
                **target_counts(row["method"], processed, post5["frames"], visible5["frames"]))


def load_native(directory, workspace):
    selection = read(directory / "selection.json")
    cases = {c["case_id"]: c for c in selection["cases"]}
    visibility, annotations = {}, {}
    for key, case in cases.items():
        folder = Path(case["annotation_dir"].replace("${WORKSPACE_ROOT}", str(workspace))
                      .replace("${TEST10_ROOT}", str(ROOT)))
        for offset in range(1, 6):
            position = case["switch"] + offset
            if position > case["end"]:
                continue
            path = folder / (case["frame_stems"][position] + ".png")
            if path.is_file():
                with Image.open(path) as image:
                    labels = np.asarray(image)
                if labels.ndim != 2 or case["object_id"] in (0, 255):
                    raise ValueError(f"invalid indexed object annotation: {path}")
                visibility[key, offset] = bool((labels == case["object_id"]).any())
                annotations[str(path)] = sha(path)
    rows = []
    original = read(directory / "case_scores.json")
    for row in original:
        if row["status"] != "complete":
            continue
        case, frames = cases[row["case_id"]], []
        for offset in range(1, 6):
            frame = row["scores"]["switch_frames"][f"+{offset}"]
            if frame["status"] != "scored" or (row["case_id"], offset) not in visibility:
                raise ValueError("first-five metrics/annotations missing; refuse to shorten window")
            if frame["position"] != case["switch"] + offset or frame["frame_stem"] != case["frame_stems"][frame["position"]]:
                raise ValueError("first-five timeline mismatch")
            frames.append(dict(frame, annotated=True, gt_present=visibility[row["case_id"], offset]))
        rows.append(normalized(row, frames, case["end"] - case["switch"], row["scores"], case["first"]))
    groups = complete_groups(rows, selection["methods"])
    return [r for g in groups.values() for r in g.values()], dict(
        source_files={str(directory / f): sha(directory / f) for f in ("selection.json", "case_scores.json")},
        first5_annotations_sha256=annotations, expected_cases=len(cases), completed_common_cases=len(groups),
        nonzero_prompt_cases=[c["case_id"] for c in cases.values() if c["first"] != 0],
        window="Full suffix available; early-window metrics use processed offsets +1..+5",
        warning="Warm Target is native from original prompt, not always frame 0. DAVIS train is exploratory; "
                "some videos appeared in earlier evaluations, so this is not an untouched test." )


def load_bank(directory):
    # Fix the file list before reading. Atomic upstream writes make this safe during inference.
    files = sorted((directory / "scores").glob("*/*.json"))
    provenance = read(directory / "provenance.json")
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rows, evidence, identities = [], {}, {}
    for path in files:
        row = read(path)
        if row["schema"] != "test10.downloaded_bank_first5.v1" or row["fingerprint"] != fingerprint:
            raise ValueError(f"unverified bank row: {path}")
        frames = row["scores"]["frames"]
        if row["horizon"] != 5 or len(frames) != 5 or [f["offset"] for f in frames] != list(range(1, 6)):
            raise ValueError(f"unexpected bank horizon: {path}")
        if any(not f["annotated"] or f["frame"] != row["switch"] + f["offset"] for f in frames):
            raise ValueError(f"bank GT/timeline mismatch: {path}")
        if row["runtime"]["past_backbone_calls"] != 0:
            raise ValueError("unexpected replay in no-replay method")
        identity = [row[k] for k in ("dataset", "video_id", "object_id", "switch", "frame_stems", "pair_sha256", "input_fingerprint", "evaluation_role")]
        identity.append([f["gt_present"] for f in frames])
        if row["pair_id"] in identities and identities[row["pair_id"]] != identity:
            raise ValueError("method input conditions differ")
        identities[row["pair_id"]] = identity
        evidence[str(path.relative_to(directory))] = sha(path)
        rows.append(normalized(row, frames, 5, first=None))
    groups = complete_groups(rows, BANK_METHODS)
    audit = read(directory / "audit.json")
    return [r for g in groups.values() for r in g.values()], dict(
        provenance_sha256=sha(directory / "provenance.json"), score_files_sha256=evidence,
        expected_cases=audit["total_pairs"], completed_common_cases=len(groups),
        excluded_incomplete_cases=len(identities) - len(groups),
        window="Only processed offsets +1..+5 available; GT-visible is GT-visible@5, not full suffix",
        warning="Running fixed-order partial snapshot, not final full-bank result. No Warm Target predictions: "
                "all retention metrics unavailable. All-pool contains fitting/model-selection videos; "
                "video-disjoint cohort is the primary analysis. Do not combine with native-prefix experiment.")


def video_values(rows, metric, field="J_and_F"):
    videos = defaultdict(list)
    for row in rows:
        value = row["metrics"][metric][field]
        if value is not None:
            videos[row["video_id"]].append(value)
    return {v: mean(values) for v, values in videos.items()}


def matched_retention(rows, method, metric):
    a = video_values([r for r in rows if r["method"] == method], metric)
    b = video_values([r for r in rows if r["method"] == "base_native"], metric)
    keys = sorted(a.keys() & b.keys())
    return dict(percent=retention(mean(a[k] for k in keys), mean(b[k] for k in keys)) if keys else None,
                matched_videos=len(keys), denominator="base_native", aggregation="ratio of matched video means")


def summaries(rows, cohort):
    output = []
    for dataset in sorted({r["dataset"] for r in rows}):
        dataset_rows = [r for r in rows if r["dataset"] == dataset]
        for method in sorted({r["method"] for r in dataset_rows}):
            selected = [r for r in dataset_rows if r["method"] == method]
            out = dict(cohort=cohort, dataset=dataset, method=method,
                       cases=len(selected), videos=len({r["video_id"] for r in selected}))
            for metric in ("post5", "visible5", "post_available", "visible_available"):
                values = video_values(selected, metric)
                out[metric] = dict(videos=len(values), frames=sum(r["metrics"][metric]["frames"] for r in selected),
                                  **{field: 100 * mean(video_values(selected, metric, field).values()) if values else None for field in FIELDS})
                out["retention_" + metric] = matched_retention(dataset_rows, method, metric)
            for key in ("target_frames_processed", "target_frames_scored_at5", "target_visible_frames_scored_at5"):
                out[key] = sum(r[key] for r in selected)
            output.append(out)
    return output


def contrasts(rows):
    output = []
    rng = np.random.default_rng(7)
    for dataset in sorted({r["dataset"] for r in rows}):
        for metric in ("post5", "visible5"):
            for left, right in (("affine", "direct"), ("residual_mlp", "affine"), ("transformer", "affine")):
                subset = [r for r in rows if r["dataset"] == dataset]
                a = video_values([r for r in subset if r["method"] == left], metric)
                b = video_values([r for r in subset if r["method"] == right], metric)
                keys = sorted(a.keys() & b.keys())
                if not keys:
                    continue
                delta = np.array([100 * (a[k] - b[k]) for k in keys])
                bootstrap = np.concatenate([delta[rng.integers(len(delta), size=(200, len(delta)))].mean(axis=1) for _ in range(10)])
                output.append(dict(dataset=dataset, metric=metric, contrast=f"{left}-{right}",
                                   videos=len(keys), difference=float(delta.mean()),
                                   ci95=np.quantile(bootstrap, [.025, .975]).tolist()))
    return output


def flatten_summary(row):
    out = {k: v for k, v in row.items() if not isinstance(v, dict)}
    for metric in ("post5", "visible5", "post_available", "visible_available"):
        out.update({metric + "_" + k: v for k, v in row[metric].items()})
        out["retention_" + metric + "_percent"] = row["retention_" + metric]["percent"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, default=ROOT / "results/fit1000_eval160")
    parser.add_argument("--bank-dir", type=Path, default=ROOT / "runs/downloaded_bank_first5")
    parser.add_argument("--workspace", type=Path, default=Path("/home/home/test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists; preserve earlier snapshots")
    started = datetime.now(timezone.utc).isoformat()
    native, native_evidence = load_native(args.native_dir, args.workspace)
    bank, bank_evidence = load_bank(args.bank_dir)
    disjoint = [r for r in bank if r["evaluation_role"] == "excluded_from_training_and_selection"]
    cohorts = [("native160", native), ("native159_prompt0", [r for r in native if r["first_prompt_position"] == 0]),
               ("bank_all", bank), ("bank_video_disjoint", disjoint)]
    summary = [s for name, rows in cohorts for s in summaries(rows, name)]
    report = dict(schema="test10.requested_metrics.v1", snapshot_started_utc=started,
        generated_utc=datetime.now(timezone.utc).isoformat(), gpu_inference=False, training=False,
        definitions=dict(post5="Mean over annotated processed offsets +1..+5; includes GT-absent frames",
            visible5="Same window, exclude GT-absent frames; omit cases/videos with no visible frame",
            visible_available="Full post-switch suffix for native160; ONLY first five for bank",
            retention="100 * method video-mean J&F / Warm Target video-mean J&F, same cases and window",
            target_frames="Separate actual processed target suffix frames from first-five scored target frames. Source Only=0",
            aggregation="frame mean within case, case mean within video, equal mean over videos; points 0..100",
            ci="Paired video bootstrap, 2000 draws, seed 7, pointwise without multiplicity correction"),
        native_evidence=native_evidence, bank_evidence=bank_evidence, summaries=summary,
        paired_contrasts={name: contrasts(rows) for name, rows in cohorts if name != "bank_all"},
        unavailable_methods=["First Only / Reset", "First + Last", "Norm-Matched Copy", "Replay-4", "Replay-8"],
        warning="Warm Target is an empirical reference, not a mathematical upper bound; retention can exceed 100%. "
                "Bank target tensors do not supply true native prediction scores. Old method/checkpoint results are not merged.")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    flat = [flatten_summary(r) for r in summary]
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    with (args.output_dir / "native_case_metrics.json").open("w") as stream:
        json.dump(native, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(snapshot=started, output=str(args.output_dir), native_cases=len(native)//8,
                         bank_common_cases=len(bank)//4, bank_disjoint_cases=len(disjoint)//4), indent=2))


if __name__ == "__main__":
    main()
