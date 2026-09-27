from __future__ import annotations

import pytest

from vos_memory_inspector.temporal_evaluation import evaluate_temporal_handoff


def _report(scores: list[float], present: list[bool] | None = None) -> dict:
    present = [True] * len(scores) if present is None else present
    return {
        "frames": [
            {
                "frame": index,
                "ground_truth_present": visible,
                "J_and_F": score,
            }
            for index, (score, visible) in enumerate(
                zip(scores, present, strict=True), start=11
            )
        ]
    }


def test_temporal_metrics_capture_shock_break_and_recovery() -> None:
    reference = _report([0.8] * 8)
    candidate = _report([0.0, 0.0, 0.7, 0.76, 0.78, 0.79, 0.8, 0.8])

    result = evaluate_temporal_handoff(
        candidate,
        reference,
        switch_frame=10,
        checkpoints=(1, 5),
    )

    assert result["checkpoint_scores"]["1"]["candidate_J_and_F"] == 0.0
    assert result["checkpoint_scores"]["5"]["reference_gap"] == pytest.approx(0.02)
    assert result["switch_shock"]["windows"]["1"]["mean_reference_gap"] == 0.8
    assert result["identity_break_proxy"]["occurred"] is True
    assert result["identity_break_proxy"]["runs"][0] == {
        "start_frame": 11,
        "end_frame": 12,
        "visible_frames": 2,
    }
    assert result["recovery"]["recovered"] is True
    assert result["recovery"]["recovery_frame"] == 14
    assert result["recovery"]["confirmation_frame"] == 16
    assert result["recovery"]["frames_from_switch"] == 4


def test_temporal_metrics_ignore_absent_gt_and_report_censoring() -> None:
    reference = _report([1.0, 0.8, 0.8, 0.8], [False, True, True, True])
    candidate = _report([1.0, 0.0, 0.0, 0.3], [False, True, True, True])

    result = evaluate_temporal_handoff(
        candidate,
        reference,
        switch_frame=10,
        checkpoints=(1, 5),
    )

    assert result["checkpoint_scores"]["1"]["ground_truth_present"] is False
    assert result["checkpoint_scores"]["5"]["available"] is False
    assert result["switch_shock"]["windows"]["1"]["mean_reference_gap"] == 0.8
    assert result["identity_break_proxy"]["occurred"] is True
    assert result["recovery"]["recovered"] is False
    assert result["recovery"]["frames_from_switch"] is None
    assert result["recovery"]["censored_at_frame"] == 14


def test_temporal_metrics_reject_misaligned_frames() -> None:
    candidate = _report([0.5, 0.5])
    reference = _report([0.5])
    with pytest.raises(ValueError, match="frame ids differ"):
        evaluate_temporal_handoff(candidate, reference, switch_frame=10)
