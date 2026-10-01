#!/usr/bin/env python3
"""CPU-only matched-method early/full/tail comparison from existing frame scores."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("small_only", "base_native", "direct", "affine", "residual_mlp", "transformer", "last_mask", "anchor_replay_16")
DATASETS = ("MOSEv2", "LVOSv2", "DAVIS2017", "VOST")
WINDOWS = ("first5", "full_suffix", "last5", "last_frame", "offset1", "offset2", "offset3", "offset4", "offset5")


def positions(case, window):
    if window == "full_suffix":
        return range(case["switch"] + 1, case["end"] + 1)
    if window == "first5":
        return range(case["switch"] + 1, min(case["switch"] + 5, case["end"]) + 1)
    if window == "last5":
        return range(max(case["switch"] + 1, case["end"] - 4), case["end"] + 1)
    if window == "last_frame":
        return [case["end"]]
    return [case["switch"] + int(window.removeprefix("offset"))]


def bootstrap(values):
    if not values:
        return None
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(7)
    means = np.concatenate([values[rng.integers(len(values), size=(250, len(values)))].mean(axis=1) for _ in range(20)])
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist(), videos=len(values))


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "results/paper_metrics_20261001")
    parser.add_argument("--native-dir", type=Path, default=ROOT / "results/fit1000_eval160")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("do not overwrite previous exports")
    selection = json.loads((args.native_dir / "selection.json").read_text())
    cases = {c["case_id"]: c for c in selection["cases"] if c["first"] == 0}
    if len({(c["dataset"], c["video_id"]) for c in cases.values()}) != len(cases):
        raise ValueError("expected exactly one case per video")
    frames = defaultdict(dict)
    with (args.paper_dir / "visible_frame_scores.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in cases and row["gt_present"] == "True":
                frames[row["case_id"], row["method"]][int(row["position"])] = {f: float(row[f]) for f in ("J", "F", "J_and_F")}
    values, counts = {}, {}
    for cid, case in cases.items():
        for window in WINDOWS:
            target_positions = set(positions(case, window))
            eligible = target_positions & frames[cid, "base_native"].keys()
            for method in METHODS:
                if target_positions & frames[cid, method].keys() != eligible:
                    raise ValueError("different scored frames between methods")
                counts[cid, window] = len(eligible)
                values[cid, method, window] = {field: 100 * mean(frames[cid, method][p][field] for p in eligible) if eligible else None
                                              for field in ("J", "F", "J_and_F")}
    summary, differences, drift_change = [], [], []
    for dataset in DATASETS:
        primary = "J" if dataset == "VOST" else "J_and_F"
        ids = [cid for cid, case in cases.items() if case["dataset"] == dataset]
        for window in WINDOWS:
            eligible = [cid for cid in ids if counts[cid, window]]
            for method in METHODS:
                row = dict(dataset=dataset, method=method, window=window, primary_metric=primary,
                    videos=len(eligible), visible_frames=sum(counts[cid, window] for cid in eligible))
                row.update({field: mean(values[cid, method, window][field] for cid in eligible) if eligible else None for field in ("J", "F", "J_and_F")})
                summary.append(row)
                if method == "affine":
                    continue
                delta = [values[cid, "affine", window][primary] - values[cid, method, window][primary] for cid in eligible]
                stat = bootstrap(delta)
                if stat:
                    differences.append(dict(dataset=dataset, window=window, contrast="affine-minus-" + method,
                        **{k: v for k, v in stat.items() if k != "ci95"}, ci95_low=stat["ci95"][0], ci95_high=stat["ci95"][1]))
        for method in ("affine", "residual_mlp", "transformer"):
            for late in ("full_suffix", "last5"):
                eligible = [cid for cid in ids if counts[cid, "first5"] and counts[cid, late]]
                delta = [(values[cid, "base_native", late][primary] - values[cid, method, late][primary]) -
                         (values[cid, "base_native", "first5"][primary] - values[cid, method, "first5"][primary]) for cid in eligible]
                stat = bootstrap(delta)
                if stat:
                    drift_change.append(dict(dataset=dataset, method=method, late_window=late,
                        definition="Same-video change of native-minus-method gap relative to first5; positive means gap larger",
                        **stat))
    # Reaggregation must reproduce the previous full-suffix quality report.
    with (args.paper_dir / "quality.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["cohort"] != "native159_prompt0":
                continue
            computed = next(r for r in summary if r["dataset"] == row["dataset"] and r["method"] == row["method"] and r["window"] == "full_suffix")
            for field in ("J", "F", "J_and_F"):
                if abs(computed[field] - float(row[field])) > 1e-10:
                    raise ValueError("full-suffix score parity failed")
    report = dict(schema="test10.affine_windows.v1", inference=False, training=False,
        cohort="native159_prompt0", cases=len(cases), quality=summary, paired_differences=differences,
        matched_gap_change=drift_change,
        warning="GT-visible only; support changes by window, offsets are processed sampled frames. "
                "MOSE train/development and DAVIS train/exploratory, not official scores. "
                "CI: paired-video bootstrap, 5000 draws, seed 7; pointwise, no multiplicity correction. "
                "Window-relative gaps are not a proof of causal time deterioration.")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    save_csv(args.output_dir / "windows.csv", summary)
    save_csv(args.output_dir / "affine_differences.csv", differences)
    print(json.dumps(dict(output=str(args.output_dir), cases=len(cases), full_suffix_parity=True)))


if __name__ == "__main__":
    main()
