#!/usr/bin/env python3
"""Run the fixed Phase-1 SAM 2 baseline ladder on one prepared case cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vos_memory_inspector.davis_evaluation import (
    evaluate_davis_future_masks,
    load_official_davis_metrics,
    write_davis_future_report,
)
from vos_memory_inspector.roundtrip import (
    run_cached_baseline,
    run_cached_translator_handoff,
)
from vos_memory_inspector.temporal_evaluation import evaluate_temporal_handoff


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Direct, Reset, Last-Mask, Replay-1/2/4 and Full Replay on one "
            "checksummed prepared case, then evaluate partial DAVIS J&F."
        )
    )
    parser.add_argument("--case-cache", required=True, type=Path)
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--annotation-dir", required=True, type=Path)
    parser.add_argument("--evaluation-repo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _metric_row(
    method_id: str,
    report: dict[str, Any],
    davis: dict[str, Any],
) -> dict[str, Any]:
    resources = report["resources_candidate_only"]
    if "baseline_plan" in report:
        plan = report["baseline_plan"]
        history_frames = int(plan["history_frames_reprocessed"])
        prompt_source = plan["prompt_source"]
        uses_gt = bool(plan["uses_ground_truth_prompt"])
        past_calls = int(report["backbone_calls_before_or_at_switch_total"])
    else:
        history_frames = 0
        prompt_source = "translated_source_state"
        uses_gt = False
        past_calls = int(report["backbone_calls_before_injection"]) + int(
            report["backbone_calls_during_injection"]
        )
    comparison = report["mask_comparison_to_target_native"]
    return {
        "method": method_id,
        "label": report.get("baseline_label", "Direct Copy"),
        "prompt_source": prompt_source,
        "uses_ground_truth_prompt": uses_gt,
        "history_frames_reprocessed": history_frames,
        "mean_J": davis["mean_J"],
        "mean_F": davis["mean_F"],
        "mean_J_and_F": davis["mean_J_and_F"],
        "ground_truth_visible_frames": davis["ground_truth_visible"]["frames"],
        "mean_visible_J_and_F": davis["ground_truth_visible"]["mean_J_and_F"],
        "ground_truth_absent_frames": davis["ground_truth_absent"]["frames"],
        "mean_absent_J_and_F": davis["ground_truth_absent"]["mean_J_and_F"],
        "mean_binary_iou_to_target_native": comparison["mean_binary_iou"],
        "wall_time_seconds": resources["wall_time_seconds"],
        "peak_cuda_memory_bytes": resources["peak_cuda_memory_bytes"],
        "backbone_calls_before_or_at_switch": past_calls,
        "backbone_calls_during_future": int(
            report["backbone_calls_during_future_continuation"]
        ),
    }


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Cached SAM 2 baseline suite",
        "",
        "> One video/object/switch partial DAVIS evaluation; not a full benchmark.",
        "",
        f"- Sequence: `{summary['sequence']}`",
        f"- Object: `{summary['object_id']}`",
        f"- Switch frame: `{summary['switch_frame']}`",
        f"- Evaluated frames: `{summary['start_frame']}–{summary['end_frame']}`",
        f"- Last-Mask = Replay-1 sanity check: `{summary['sanity_checks']['last_mask_equals_replay_1']}`",
        "",
        "| Method | Prompt/input | Prefix frames | J&F | Visible J&F | Native IoU | Wall s | Peak GiB | Past backbone | Future backbone |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["methods"]:
        peak = row["peak_cuda_memory_bytes"]
        peak_gib = "n/a" if peak is None else f"{peak / (1024 ** 3):.3f}"
        visible_jf = row["mean_visible_J_and_F"]
        visible_jf_text = "n/a" if visible_jf is None else f"{visible_jf:.6f}"
        lines.append(
            "| {label} | `{prompt_source}` | {history_frames_reprocessed} | "
            "{mean_J_and_F:.6f} | {visible_jf_text} | {mean_binary_iou_to_target_native:.6f} | "
            "{wall_time_seconds:.3f} | {peak_gib} | "
            "{backbone_calls_before_or_at_switch} | {backbone_calls_during_future} |".format(
                peak_gib=peak_gib, visible_jf_text=visible_jf_text, **row
            )
        )
    lines.extend(
        (
            "",
            "## Post-switch temporal metrics",
            "",
            "| Method | +1 J&F | +5 J&F | +20 J&F | Shock first 5 visible | Identity-loss proxy | Recovery frames |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in summary["methods"]:
        temporal = row["temporal"]
        checkpoints = temporal["checkpoint_scores"]

        def checkpoint_text(offset: str) -> str:
            value = checkpoints[offset]["candidate_J_and_F"]
            return "n/a" if value is None else f"{value:.6f}"

        shock = temporal["switch_shock"]["windows"]["5"]["mean_reference_gap"]
        shock_text = "n/a" if shock is None else f"{shock:.6f}"
        identity = "yes" if temporal["identity_break_proxy"]["occurred"] else "no"
        recovery = temporal["recovery"]["frames_from_switch"]
        recovery_text = "censored" if recovery is None else str(recovery)
        lines.append(
            f"| {row['label']} | {checkpoint_text('1')} | {checkpoint_text('5')} | "
            f"{checkpoint_text('20')} | {shock_text} | {identity} | {recovery_text} |"
        )
    lines.extend(
        (
            "",
            "`target_reset` is a SAM 2 runtime proxy: an all-zero mask registers the "
            "object on the switch frame, but no source object information or temporal "
            "memory is transferred.",
            "",
            "Visible J&F averages only frames where the selected object exists in "
            "DAVIS GT. It prevents empty-GT/empty-prediction frames from making a "
            "failed reappearance look successful.",
            "",
        )
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "case_cache": args.case_cache,
        "sam2_repo": args.sam2_repo,
        "target_config_file": args.target_config,
        "target_checkpoint": args.target_checkpoint,
        "target_model_id": args.target_model_id,
        "video_dir": args.video_dir,
        "annotation_dir": args.annotation_dir,
        "device": args.device,
        "offload_video_to_cpu": not args.keep_video_on_device,
        "offload_state_to_cpu": not args.keep_state_on_device,
        "seed": args.seed,
    }
    runs: list[tuple[str, dict[str, Any], Path]] = []

    direct_dir = output_dir / "direct_copy"
    direct = run_cached_translator_handoff(
        **common,
        artifact_dir=direct_dir,
    )
    runs.append(("direct_copy", direct, direct_dir))

    definitions = [
        ("target_reset", None),
        ("last_mask", None),
        ("replay_k", 1),
        ("replay_k", 2),
        ("replay_k", 4),
        ("full_replay", None),
    ]
    reports: dict[str, dict[str, Any]] = {}
    for baseline, replay_frames in definitions:
        method_id = (
            f"replay_{replay_frames}" if baseline == "replay_k" else baseline
        )
        method_dir = output_dir / method_id
        report = run_cached_baseline(
            **common,
            baseline=baseline,
            replay_frames=replay_frames,
            prompt_mask=args.prompt_mask if baseline == "full_replay" else None,
            artifact_dir=method_dir,
        )
        reports[method_id] = report
        runs.append((method_id, report, method_dir))

    last_comparison = reports["last_mask"]["mask_comparison_to_target_native"]
    replay_one_comparison = reports["replay_1"][
        "mask_comparison_to_target_native"
    ]
    if last_comparison != replay_one_comparison:
        raise RuntimeError("Last-Mask and Replay-1 produced different results")

    iou_metric, boundary_metric, evaluator_commit = load_official_davis_metrics(
        args.evaluation_repo
    )
    method_rows: list[dict[str, Any]] = []
    davis_reports: dict[str, dict[str, Any]] = {}
    for method_id, report, method_dir in runs:
        frames = [
            int(row["frame"])
            for row in report["mask_comparison_to_target_native"]["frames"]
        ]
        start_frame = min(frames)
        end_frame = max(frames) - 1
        davis = evaluate_davis_future_masks(
            prediction_directory=method_dir / "candidate_masks",
            annotation_directory=args.annotation_dir,
            object_id=int(report["object_id"]),
            start_frame=start_frame,
            end_frame=end_frame,
            iou_metric=iou_metric,
            boundary_metric=boundary_metric,
            metric_source=(
                "davisvideochallenge/davis2017-evaluation@" + evaluator_commit
            ),
            sequence=report["video_id"],
        )
        write_davis_future_report(davis, method_dir / "davis.json")
        davis_reports[method_id] = davis
        method_rows.append(_metric_row(method_id, report, davis))

    reference = davis_reports["full_replay"]
    for row in method_rows:
        row["temporal"] = evaluate_temporal_handoff(
            davis_reports[row["method"]],
            reference,
            switch_frame=int(runs[0][1]["switch_frame"]),
        )

    first = runs[0][1]
    summary = {
        "schema_version": "cmmt.cached_baseline_suite.v2",
        "scope": "one_video_object_switch_partial_davis",
        "warning": "This is not a full DAVIS benchmark result.",
        "sequence": first["video_id"],
        "object_id": int(first["object_id"]),
        "switch_frame": int(first["switch_frame"]),
        "start_frame": min(
            int(row["frame"])
            for row in first["mask_comparison_to_target_native"]["frames"]
        ),
        "end_frame": max(
            int(row["frame"])
            for row in first["mask_comparison_to_target_native"]["frames"]
        )
        - 1,
        "official_evaluator_commit": evaluator_commit,
        "sanity_checks": {"last_mask_equals_replay_1": True},
        "methods": method_rows,
    }
    _write_summary(output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
