"""Post-switch temporal metrics for single-object VOS handoff evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any


def _frame_rows(report: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = {int(row["frame"]): row for row in report["frames"]}
    if len(rows) != len(report["frames"]):
        raise ValueError("temporal report contains duplicate frame ids")
    return rows


def _mean_or_none(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def evaluate_temporal_handoff(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    switch_frame: int,
    checkpoints: Sequence[int] = (1, 5, 20),
    recovery_tolerance: float = 0.05,
    recovery_consecutive_visible: int = 3,
    reference_healthy_threshold: float = 0.5,
    identity_failure_threshold: float = 0.1,
    identity_failure_consecutive_visible: int = 2,
) -> dict[str, Any]:
    """Compare candidate behavior with a target-native reference after handoff.

    True identity switches require multi-object correspondence.  For the current
    single-object DAVIS protocol we therefore report an explicit loss proxy:
    consecutive visible observations where the reference tracks the object but
    the candidate score is near zero.
    """

    if recovery_consecutive_visible < 1:
        raise ValueError("recovery_consecutive_visible must be positive")
    if identity_failure_consecutive_visible < 1:
        raise ValueError("identity_failure_consecutive_visible must be positive")
    if recovery_tolerance < 0:
        raise ValueError("recovery_tolerance must be non-negative")

    candidate_rows = _frame_rows(candidate)
    reference_rows = _frame_rows(reference)
    if set(candidate_rows) != set(reference_rows):
        raise ValueError("candidate and reference frame ids differ")
    frames = sorted(frame for frame in candidate_rows if frame > switch_frame)
    if not frames:
        raise ValueError("no post-switch frames are available")

    checkpoint_rows: dict[str, dict[str, Any]] = {}
    for offset in checkpoints:
        if offset < 1:
            raise ValueError("checkpoint offsets must be positive")
        frame = switch_frame + int(offset)
        candidate_row = candidate_rows.get(frame)
        reference_row = reference_rows.get(frame)
        if candidate_row is None or reference_row is None:
            checkpoint_rows[str(offset)] = {
                "frame": frame,
                "available": False,
                "ground_truth_present": None,
                "candidate_J_and_F": None,
                "reference_J_and_F": None,
                "reference_gap": None,
            }
            continue
        candidate_score = float(candidate_row["J_and_F"])
        reference_score = float(reference_row["J_and_F"])
        checkpoint_rows[str(offset)] = {
            "frame": frame,
            "available": True,
            "ground_truth_present": bool(candidate_row["ground_truth_present"]),
            "candidate_J_and_F": candidate_score,
            "reference_J_and_F": reference_score,
            "reference_gap": reference_score - candidate_score,
        }

    visible = [
        (
            frame,
            float(candidate_rows[frame]["J_and_F"]),
            float(reference_rows[frame]["J_and_F"]),
        )
        for frame in frames
        if bool(candidate_rows[frame]["ground_truth_present"])
    ]
    shock_windows: dict[str, dict[str, Any]] = {}
    for window in checkpoints:
        rows = visible[: int(window)]
        candidate_scores = [row[1] for row in rows]
        reference_scores = [row[2] for row in rows]
        shock_windows[str(window)] = {
            "requested_visible_frames": int(window),
            "evaluated_visible_frames": len(rows),
            "candidate_mean_J_and_F": _mean_or_none(candidate_scores),
            "reference_mean_J_and_F": _mean_or_none(reference_scores),
            "mean_reference_gap": _mean_or_none(
                [reference - candidate for candidate, reference in zip(
                    candidate_scores, reference_scores, strict=True
                )]
            ),
        }

    bad_runs: list[dict[str, int]] = []
    run_start: int | None = None
    run_frames: list[int] = []
    for frame, candidate_score, reference_score in visible:
        failed = (
            reference_score >= reference_healthy_threshold
            and candidate_score <= identity_failure_threshold
        )
        if failed:
            if run_start is None:
                run_start = frame
            run_frames.append(frame)
        else:
            if len(run_frames) >= identity_failure_consecutive_visible:
                bad_runs.append(
                    {
                        "start_frame": int(run_start),
                        "end_frame": run_frames[-1],
                        "visible_frames": len(run_frames),
                    }
                )
            run_start = None
            run_frames = []
    if len(run_frames) >= identity_failure_consecutive_visible:
        bad_runs.append(
            {
                "start_frame": int(run_start),
                "end_frame": run_frames[-1],
                "visible_frames": len(run_frames),
            }
        )

    recovery_start: int | None = None
    recovery_confirmation: int | None = None
    stable: list[int] = []
    for frame, candidate_score, reference_score in visible:
        recovered = (
            reference_score >= reference_healthy_threshold
            and reference_score - candidate_score <= recovery_tolerance
        )
        if recovered:
            stable.append(frame)
            if len(stable) >= recovery_consecutive_visible:
                recovery_start = stable[-recovery_consecutive_visible]
                recovery_confirmation = frame
                break
        else:
            stable = []

    return {
        "schema_version": "cmmt.temporal_handoff_evaluation.v1",
        "reference": "target_native_full_replay",
        "switch_frame": int(switch_frame),
        "checkpoint_scores": checkpoint_rows,
        "switch_shock": {
            "definition": (
                "target-native minus candidate mean J&F over the first N "
                "GT-visible post-switch observations"
            ),
            "windows": shock_windows,
        },
        "identity_break_proxy": {
            "definition": (
                "single-object proxy: candidate J&F <= threshold while the "
                "target-native reference remains healthy"
            ),
            "candidate_failure_threshold": identity_failure_threshold,
            "reference_healthy_threshold": reference_healthy_threshold,
            "required_consecutive_visible": identity_failure_consecutive_visible,
            "occurred": bool(bad_runs),
            "runs": bad_runs,
        },
        "recovery": {
            "definition": (
                "first of consecutive GT-visible observations whose candidate "
                "J&F is no more than tolerance below a healthy target-native reference"
            ),
            "tolerance": recovery_tolerance,
            "reference_healthy_threshold": reference_healthy_threshold,
            "required_consecutive_visible": recovery_consecutive_visible,
            "recovered": recovery_start is not None,
            "recovery_frame": recovery_start,
            "confirmation_frame": recovery_confirmation,
            "frames_from_switch": (
                None if recovery_start is None else recovery_start - switch_frame
            ),
            "censored_at_frame": None if recovery_start is not None else frames[-1],
        },
    }
