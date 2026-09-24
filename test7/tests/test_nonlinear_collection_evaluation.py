from __future__ import annotations

import pytest

from scripts.evaluate_nonlinear_collection import _aggregate_rows


def _row(
    *,
    visible_frames: int,
    visible_jf: float | None,
    absent_jf: float,
    shock: float | None,
    identity: bool,
    recovered: bool,
    wall: float,
) -> dict:
    return {
        "candidate_davis": {
            "ground_truth_visible": {
                "frames": visible_frames,
                "mean_J_and_F": visible_jf,
            },
            "ground_truth_absent": {"mean_J_and_F": absent_jf},
        },
        "temporal": {
            "switch_shock": {"windows": {"5": {"mean_reference_gap": shock}}},
            "identity_break_proxy": {"occurred": identity},
            "recovery": {"recovered": recovered},
        },
        "handoff": {"resources_candidate_only": {"wall_time_seconds": wall}},
    }


def test_aggregate_excludes_absent_only_cases_from_identity_and_recovery() -> None:
    rows = [
        _row(
            visible_frames=4,
            visible_jf=0.2,
            absent_jf=1.0,
            shock=0.4,
            identity=True,
            recovered=False,
            wall=2.0,
        ),
        _row(
            visible_frames=3,
            visible_jf=0.6,
            absent_jf=0.8,
            shock=0.2,
            identity=False,
            recovered=True,
            wall=4.0,
        ),
        _row(
            visible_frames=0,
            visible_jf=None,
            absent_jf=0.9,
            shock=None,
            identity=False,
            recovered=False,
            wall=3.0,
        ),
    ]

    aggregate = _aggregate_rows(rows)

    assert aggregate["case_count"] == 3
    assert aggregate["gt_visible_case_count"] == 2
    assert aggregate["gt_absent_only_case_count"] == 1
    assert aggregate["mean_gt_visible_J_and_F"] == pytest.approx(0.4)
    assert aggregate["mean_gt_absent_only_J_and_F"] == pytest.approx(0.9)
    assert aggregate["mean_switch_shock_first_5_visible"] == pytest.approx(0.3)
    assert aggregate["identity_break_proxy_rate"] == 0.5
    assert aggregate["recovery_rate"] == 0.5
    assert aggregate["mean_candidate_wall_time_seconds"] == 3.0
