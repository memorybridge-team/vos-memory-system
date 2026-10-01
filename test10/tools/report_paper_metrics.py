#!/usr/bin/env python3
"""Paper-protocol audit and CPU-only reaggregation of existing predictions.

Never generates predictions, states, annotations, benchmark submissions or weights.
Missing data remains missing. Official MOSE validation results are not synthesized
from local development scores. Ratio-zero handling is explicit and provisional.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
from statistics import mean, median
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import Artifacts, read, sha
from run import restore_selection_paths
import report_requested_metrics as requested

METHODS = ("small_only", "base_native", "direct", "affine", "residual_mlp", "transformer", "last_mask", "anchor_replay_16")
FIELD_NAMES = ("J", "F", "J_and_F")


def ratio(a, b):
    return None if a is None or b is None or b <= 0 else 100 * a / b


def gap_ratio(a, source, warm):
    if any(v is None for v in (a, source, warm)) or warm == source:
        return None
    return 100 * (a - source) / (warm - source)


def ratio_summary(values):
    valid = [v for v in values if v is not None]
    return dict(mean=mean(valid) if valid else None, median=median(valid) if valid else None,
                valid_cases=len(valid), undefined_cases=len(values) - len(valid),
                minimum=min(valid) if valid else None, maximum=max(valid) if valid else None)


def iou(a, b, void=None):
    keep = np.ones_like(a, dtype=bool) if void is None else ~void
    union = np.count_nonzero((a | b) & keep)
    return 1. if not union else np.count_nonzero((a & b) & keep) / union


def is_id_swap(prediction, labels, object_id):
    void = labels == 255
    target_iou = iou(prediction, labels == object_id, void)
    others = [int(k) for k in np.unique(labels) if k not in (0, 255, object_id)]
    return any(iou(prediction, labels == k, void) > target_iou for k in others)


def length_bin(prefix_length):
    return "1-16" if prefix_length <= 16 else "17-64" if prefix_length <= 64 else "65-256" if prefix_length <= 256 else "257+"


def drift_bin(offset):
    return "1-5" if offset <= 5 else "6-10" if offset <= 10 else "11-20" if offset <= 20 else "21-50" if offset <= 50 else "51-100" if offset <= 100 else "101+"


def mean_fields(frames):
    return {k: mean(f[k] for f in frames) if frames else None for k in FIELD_NAMES}


def boundary_context(gt, void):
    """Reuse the exact DAVIS GT boundary/dilation across same-frame methods."""
    import cv2
    from skimage.morphology import disk
    from vos_memory_inspector._vendor.davis2017_metrics import _seg2bmap
    boundary = _seg2bmap(gt & ~void)
    radius = int(np.ceil(.008 * np.linalg.norm(gt.shape)))
    kernel = disk(radius).astype(np.uint8)
    return boundary, cv2.dilate(boundary.astype(np.uint8), kernel), kernel, int(boundary.sum())


def boundary_f(pred, void, context):
    import cv2
    from vos_memory_inspector._vendor.davis2017_metrics import _seg2bmap
    gt_boundary, gt_dilated, kernel, n_gt = context
    pred_boundary = _seg2bmap(pred & ~void)
    n_pred = int(pred_boundary.sum())
    if n_pred == 0:
        return 1. if n_gt == 0 else 0.
    if n_gt == 0:
        return 0.
    pred_dilated = cv2.dilate(pred_boundary.astype(np.uint8), kernel)
    precision = np.count_nonzero(pred_boundary & (gt_dilated != 0)) / n_pred
    recall = np.count_nonzero(gt_boundary & (pred_dilated != 0)) / n_gt
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.


def cached_case(task):
    """Read verified local masks and GT. No predictor construction or CUDA calls."""
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1) if not getattr(cached_case, "threads_set", False) else None
    cached_case.threads_set = True
    import cv2
    cv2.setNumThreads(1)
    repo = Path("/home/home/test/test9/vos-memory-translator-nonlinear")
    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(repo / "src"))
    from mvp_scoring import unpack
    case, run_dir, provenance, scores = task
    store = Artifacts(run_dir, provenance)
    masks, evidence = {}, {}
    for method in METHODS:
        name = {"small_only": "source_prefix", "base_native": "base_prefix"}.get(method, method + "_predictions")
        path = store.path(case, name, ".pt")
        marker = path.with_suffix(".sha.json")
        if not path.is_file() or not marker.is_file():
            return dict(case_id=case["case_id"], missing_masks=method)
        info = read(marker)
        if info["key"] != path.stem or sha(path) != info["sha256"]:
            raise ValueError(f"invalid mask cache: {path}")
        evidence[str(path)] = info["sha256"]
        # Local self-created, hash-verified numpy payloads, loaded on CPU only.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        masks[method] = payload["masks"]
        if set(masks[method]) != set(range(case["switch"] + 1, case["end"] + 1)):
            raise ValueError("cached suffix timeline differs")
    frames = {m: [] for m in METHODS}
    gt_hashes = {}
    for position in range(case["switch"] + 1, case["end"] + 1):
        path = Path(case["annotation_dir"]) / (case["frame_stems"][position] + ".png")
        if not path.is_file():
            continue
        with Image.open(path) as image:
            labels = np.asarray(image)
        gt_hashes[str(path)] = sha(path)
        gt, void = labels == case["object_id"], labels == 255
        native = unpack(*masks["base_native"][position])
        gt_present = bool(gt.any())
        boundaries = boundary_context(gt, void)
        areas = np.bincount(labels.ravel(), minlength=256)
        other_ids = np.flatnonzero(areas)
        other_ids = other_ids[~np.isin(other_ids, [0, 255, case["object_id"]])]
        frame_cache = {}
        for method in METHODS:
            packed = masks[method][position]
            if packed in frame_cache:
                frames[method].append(dict(frame_cache[packed]))
                continue
            pred = native if method == "base_native" else unpack(*masks[method][position])
            intersections = np.bincount(labels[pred & ~void], minlength=256)
            pred_area = int(intersections.sum())
            union = pred_area + int(areas[case["object_id"]]) - int(intersections[case["object_id"]])
            j = 1. if not union else float(intersections[case["object_id"]] / union)
            f = boundary_f(pred, void, boundaries)
            other_unions = pred_area + areas[other_ids] - intersections[other_ids]
            swap = bool(np.any(intersections[other_ids] / other_unions > j)) if gt_present else None
            frame = dict(position=position, offset=position - case["switch"],
                gt_present=gt_present, J=j, F=f, J_and_F=(j + f) / 2,
                mask_iou_with_native=iou(pred, native, void),
                id_swap=swap)
            frame_cache[packed] = frame
            frames[method].append(frame)
    rows = []
    for method in METHODS:
        visible = [f for f in frames[method] if f["gt_present"]]
        avg = mean_fields(visible)
        old = scores[method]["scores"]["gt_visible"]
        if len(visible) != old["frames"] or any(
                avg[k] is not None and abs(avg[k] - old[k]) > 1e-10 for k in FIELD_NAMES):
            raise ValueError(f"visible-score parity failed: {case['case_id']}/{method}: {avg} != {old}")
        runtime = scores[method]["runtime"]
        rows.append(dict(case_id=case["case_id"], dataset=case["dataset"], video_id=case["video_id"],
            method=method, first=case["first"], prefix_frames=case["switch"] - case["first"] + 1,
            prefix_length_bin=length_bin(case["switch"] - case["first"] + 1),
            raw_stride=case["sampling"]["raw_stride"], data_split=case.get("data_split"),
            visible_frames=len(visible), scored_frames=len(frames[method]), **avg,
            failed=avg["J"] < .1 if avg["J"] is not None else None,
            id_swap_rate=mean(f["id_swap"] for f in visible) if visible else None,
            mask_iou_with_native=mean(f["mask_iou_with_native"] for f in frames[method]) if frames[method] else None,
            visible_mask_iou_with_native=mean(f["mask_iou_with_native"] for f in visible) if visible else None,
            post_frame_seconds=runtime["continuation_s"] / (case["end"] - case["switch"]), frames=frames[method]))
    return dict(case_id=case["case_id"], rows=rows, evidence=evidence, gt_hashes=gt_hashes)


def quality_summary(rows, cohort):
    by_case = {(r["case_id"], r["method"]): r for r in rows}
    output, case_ratios = [], []
    for row in rows:
        native, source = by_case[row["case_id"], "base_native"], by_case[row["case_id"], "small_only"]
        values = {"case_id": row["case_id"], "dataset": row["dataset"], "video_id": row["video_id"],
                  "method": row["method"], "prefix_length_bin": row["prefix_length_bin"]}
        for field in ("J", "J_and_F"):
            values["recovery_" + field] = ratio(row[field], native[field])
            values["gap_recovery_" + field] = gap_ratio(row[field], source[field], native[field])
            values["warm_zero_" + field] = native[field] == 0
            values["gap_negative_" + field] = native[field] < source[field] if native[field] is not None and source[field] is not None else None
            values["gap_zero_" + field] = native[field] == source[field]
        case_ratios.append(values)
    for dataset in sorted({r["dataset"] for r in rows}):
        for method in METHODS:
            selected = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            ratios = [r for r in case_ratios if r["dataset"] == dataset and r["method"] == method]
            visible = [r for r in selected if r["J"] is not None]
            entry = dict(cohort=cohort, dataset=dataset, method=method, cases=len(selected),
                visible_cases=len(visible), visible_frames=sum(r["visible_frames"] for r in selected),
                primary_metric="J" if dataset == "VOST" else "J_and_F",
                official_final=False, scope="local development post-switch GT-visible full suffix",
                **{k: 100 * mean(r[k] for r in visible) if visible else None for k in FIELD_NAMES},
                failure_rate=mean(r["failed"] for r in visible) if visible else None,
                id_swap_rate=mean(r["id_swap_rate"] for r in visible) if visible else None,
                mask_iou_with_native=mean(r["mask_iou_with_native"] for r in selected),
                visible_mask_iou_with_native=mean(r["visible_mask_iou_with_native"] for r in visible) if visible else None,
                post_frame_seconds=mean(r["post_frame_seconds"] for r in selected))
            for field in ("J", "J_and_F"):
                for prefix in ("recovery_", "gap_recovery_"):
                    entry[prefix + field] = ratio_summary([r[prefix + field] for r in ratios])
                for prefix in ("warm_zero_", "gap_negative_", "gap_zero_"):
                    entry[prefix + field] = sum(bool(r[prefix + field]) for r in ratios)
            output.append(entry)
    return output, case_ratios


def drift_summary(rows, cohort):
    by_case = {(r["case_id"], r["method"]): r for r in rows}
    groups = defaultdict(list)
    for row in rows:
        native = {f["position"]: f for f in by_case[row["case_id"], "base_native"]["frames"]}
        buckets = defaultdict(list)
        for frame in row["frames"]:
            if frame["gt_present"]:
                buckets[drift_bin(frame["offset"])].append(frame)
        for bucket, frames in buckets.items():
            groups[row["dataset"], row["method"], bucket].append(dict(
                frames=len(frames), **{field: 100 * mean(
                    native[f["position"]][field] - f[field] for f in frames) for field in ("J", "J_and_F")}))
    return [dict(cohort=cohort, dataset=d, method=m, offset_bin=b, videos=len(values),
                 visible_frames=sum(v["frames"] for v in values),
                 native_minus_method_J=mean(v["J"] for v in values),
                 native_minus_method_J_and_F=mean(v["J_and_F"] for v in values))
            for (d, m, b), values in sorted(groups.items())]


def cost_summary(costs):
    output = []
    for method in METHODS:
        times, speeds, peaks = [], [], []
        for row in costs:
            t = row["timings"][method]
            # Warm models, cold RGB. Source Only continues rather than replaying.
            elapsed = (t["first_output_s"] - t["prefix_s"] if method == "small_only" else
                       t["first_output_s"] + t.get("export_s", 0))
            times.append(elapsed)
            speeds.append(row["timings"]["base_native"]["first_output_s"] / elapsed)
            peaks.append(t["peak_vram_bytes"])
        output.append(dict(method=method, cases=len(costs), ttff_mean_seconds=mean(times),
            ttff_median_seconds=median(times), speedup_mean_of_case_ratios=mean(speeds),
            speedup_median=median(speeds), gpu_peak_mean_MB=mean(peaks)/1e6,
            gpu_peak_max_MB=max(peaks)/1e6, gpu_peak_mean_MiB=mean(peaks)/1024**2))
    return output


def csv_file(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/fit1_eval40_fast")
    parser.add_argument("--native-dir", type=Path, default=ROOT / "results/fit1000_eval160")
    parser.add_argument("--bank-dir", type=Path, default=ROOT / "runs/downloaded_bank_first5")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("refuse to overwrite an earlier report")
    started = datetime.now(timezone.utc).isoformat()
    selection = restore_selection_paths(read(args.run_dir / "selection.json"))
    provenance = read(args.run_dir / "provenance.json")
    raw = read(args.native_dir / "case_scores.json")
    by_case = defaultdict(dict)
    for row in raw:
        by_case[row["case_id"]][row["method"]] = row
    store = Artifacts(args.run_dir, provenance)
    for case in selection["cases"]:
        for method in METHODS:
            if by_case[case["case_id"]][method]["key"] != store.path(case, method).stem:
                raise ValueError("native result and prediction-cache provenance mismatch")
    tasks = [(c, args.run_dir, provenance, by_case[c["case_id"]]) for c in selection["cases"]]
    # Fail fast before submitting a large CPU rescore if any scoring contract changed.
    cached_case(tasks[0])
    rows, evidence, missing, gt_hashes = [], {}, [], {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(cached_case, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if "missing_masks" in result:
                missing.append(result)
            else:
                rows.extend(result["rows"])
                evidence.update(result["evidence"])
                gt_hashes.update(result["gt_hashes"])
            if index % 4 == 0:
                print(f"CPU cached-mask scoring: {index}/{len(tasks)} cases", flush=True)
    rows.sort(key=lambda r: (r["case_id"], r["method"]))
    cohorts = [("native160_original_anchor", rows), ("native159_prompt0", [r for r in rows if r["first"] == 0])]
    summaries, case_ratios, curves, length_groups = [], [], [], []
    for cohort, subset in cohorts:
        quality, ratios = quality_summary(subset, cohort)
        summaries.extend(quality)
        case_ratios.extend(dict(r, cohort=cohort) for r in ratios)
        curves.extend(drift_summary(subset, cohort))
        for dataset in sorted({r["dataset"] for r in ratios}):
            for method in METHODS:
                for bucket in ("1-16", "17-64", "65-256", "257+"):
                    selected = [r for r in ratios if r["dataset"] == dataset and r["method"] == method and r["prefix_length_bin"] == bucket]
                    if selected:
                        length_groups.append(dict(cohort=cohort, dataset=dataset, method=method,
                            prefix_length_bin=bucket, cases=len(selected),
                            **{k: ratio_summary([r[k] for r in selected]) for k in ("recovery_J", "recovery_J_and_F")}))
    bank, bank_evidence = requested.load_bank(args.bank_dir)
    disjoint = [r for r in bank if r["evaluation_role"] == "excluded_from_training_and_selection"]
    bank_summaries = requested.summaries(disjoint, "stopped_bank_video_disjoint_first5")
    costs = cost_summary(read(args.native_dir / "costs.json"))
    report = dict(schema="test10.paper_metrics.v1", started_utc=started,
        generated_utc=datetime.now(timezone.utc).isoformat(), inference_performed=False, training_performed=False,
        submitted_to_server=False, status="existing-data reaggregation; NOT complete final paper benchmark",
        definitions=dict(quality="Post-switch GT-visible J/F/J&F, void label 255 excluded",
            recovery="Mean of per-case 100 * method / Base+-native, not ratio of dataset means",
            gap_recovery="Mean of per-case 100 * (method-Source Only)/(Base+-native-Source Only)",
            invalid_denominators="Warm=0 or gap=0: undefined; provisional finite-case mean with explicit coverage. No epsilon or clipping",
            quality_unit="Points 0..100", failure="Case mean visible J < 0.1; no visible frame => undefined",
            id_swap="For each visible frame: any other annotated object IoU strictly exceeds intended-object IoU; void excluded",
            drift="Mean frames within case/bin, then equal case/video mean of native-minus-method; support varies by bin",
            output_agreement="Mean per-frame binary mask IoU vs native, void excluded; both empty => 1",
            ttff="Warm loaded models, cold RGB, export + measured first-output interval; Full Replay reprocesses original prefix",
            speedup="Mean of same-case Full Replay TTFF / method TTFF over fixed 12-case cost set",
            post_frame_time="Existing continuation wall time / processed suffix frames; includes pack, and native logit hashing, so not isolated model latency",
            vram="Existing torch peak allocated bytes converted to decimal MB; not total GPU process VRAM",
            model_selection="Requested 25/50/75 recovery validation not available; frozen weights originally selected by state MSE"),
        limitations=["MOSE is local train/development, not official validation full-video score; existing F uses standard DAVIS threshold, not MOSEv2 modified F",
            "DAVIS 2017 train is exploratory; evaluation videos are disjoint from fitting/selection but most appeared in previous evaluations",
            "One independently tracked object per video; not official multi-object joint evaluation",
            "Native160 includes one nonzero prompt; native159_prompt0 explicitly excludes it",
            "Negative gap denominator means native worse than source; numeric gap ratios then lack the intended higher-is-better interpretation",
            "Small positive Warm/gap denominators can make mean per-case ratios unstable; no arbitrary caps used",
            "Stopped bank is an incomplete fixed-order first-five evaluation, lacks native/source references, masks and controlled handoff timings"],
        quality=summaries, costs=costs, drift=curves, prefix_length_groups=length_groups,
        stopped_bank=bank_summaries, bank_evidence=bank_evidence,
        existing_mask_files_sha256=evidence, annotation_files_sha256=gt_hashes,
        input_files_sha256={str(p): sha(p) for p in (args.native_dir / "case_scores.json", args.native_dir / "costs.json", args.run_dir / "selection.json", args.run_dir / "provenance.json")},
        missing_cached_masks=missing,
        requires_user_approval=["Official MOSE validation predictions/submission and full-video server metrics",
            "Full-bank Warm Target and Source Only references, full suffix predictions/timings",
            "25/50/75 switch validation on a fitting/test-disjoint video set for model selection",
            "Reset without prompt and recent-K-only without prompt: define a valid SAM2 initialization first",
            "First Only, First+Last, Norm-Matched, Replay-4/8 or additional ablations if included in the final method list",
            "Switch-B immediately before reappearance requires new switch-state generation",
            "Official attribute-label resources/definitions for conditional recovery; do not infer official labels from filenames",
            "M3VOS/PUMaVOS data and evaluation if these datasets are actually in scope"],
        primary_sources=dict(mose="https://mose.video/", mose_evaluator="https://github.com/henghuiding/MOSE-api/blob/master/MOSEv2/sam2_rcms/tools/score.py",
                             davis_metrics="https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/metrics.py"))
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    flat_quality = []
    for entry in summaries:
        flat = {k: v for k, v in entry.items() if not isinstance(v, dict)}
        for k, v in entry.items():
            if isinstance(v, dict):
                flat.update({k + "_" + field: value for field, value in v.items()})
        flat_quality.append(flat)
    csv_file(args.output_dir / "quality.csv", flat_quality)
    csv_file(args.output_dir / "case_recovery.csv", case_ratios)
    csv_file(args.output_dir / "costs.csv", costs)
    csv_file(args.output_dir / "drift.csv", curves)
    csv_file(args.output_dir / "stopped_bank_first5.csv", [requested.flatten_summary(r) for r in bank_summaries])
    with (args.output_dir / "visible_frame_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "dataset", "method", "position", "offset", "gt_present", *FIELD_NAMES, "mask_iou_with_native", "id_swap"])
        writer.writeheader()
        for row in rows:
            for frame in row["frames"]:
                writer.writerow(dict(case_id=row["case_id"], dataset=row["dataset"], method=row["method"], **frame))
    print(json.dumps(dict(output=str(args.output_dir), native_cases=len(rows)//8,
                         bank_completed=bank_evidence["completed_common_cases"], missing=missing), indent=2), flush=True)


if __name__ == "__main__":
    main()
