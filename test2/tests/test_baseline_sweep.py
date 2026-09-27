from __future__ import annotations

import json
from pathlib import Path

import pytest

from vos_memory_inspector.baseline_sweep import (
    REQUIRED_METHODS,
    aggregate_completed,
    attach_temporal_metrics,
    case_slug,
    load_complete_suite,
    select_cases,
    selection_manifest,
)


def _case(case_id: str, sequence: str, object_id: int, switch: int, tag: str) -> dict:
    return {
        "case_id": case_id,
        "sequence": sequence,
        "object_id": object_id,
        "switch_frame": switch,
        "tags": [tag],
    }


def _summary(case: dict, visible_offset: float = 0.0) -> dict:
    return {
        "sequence": case["sequence"],
        "object_id": case["object_id"],
        "switch_frame": case["switch_frame"],
        "sanity_checks": {"last_mask_equals_replay_1": True},
        "methods": [
            {
                "method": method,
                "label": method,
                "mean_J_and_F": index / 10,
                "mean_visible_J_and_F": index / 10 + visible_offset,
                "backbone_calls_before_or_at_switch": index,
                "wall_time_seconds": 20 + index,
            }
            for index, method in enumerate(REQUIRED_METHODS)
        ],
    }


def test_rare_selection_is_stable_and_checksummed() -> None:
    rare = _case("rare", "india", 3, 35, "reappearance")
    regular = _case("regular", "india", 3, 20, "regular_q50")
    manifest = {"content_sha256": "source", "cases": [rare, regular]}
    selected = select_cases(manifest)
    assert selected == [rare]
    assert case_slug(selected[0]) == "india_obj3_switch35"
    first = selection_manifest(manifest, selected, tags={"reappearance"})
    second = selection_manifest(manifest, selected, tags={"reappearance"})
    assert first == second
    assert first["case_count"] == 1
    assert len(first["content_sha256"]) == 64


def test_explicit_case_ids_override_tag_filter() -> None:
    regular = _case("regular", "demo", 1, 10, "regular_q50")
    manifest = {"content_sha256": "source", "cases": [regular]}
    assert select_cases(manifest, case_ids=["regular"]) == [regular]


def test_complete_suite_validation_and_aggregate(tmp_path: Path) -> None:
    first = _case("a", "india", 1, 38, "strong_area_drop")
    second = _case("b", "india", 3, 35, "reappearance")
    root = tmp_path / "suites"
    for case, offset in ((first, 0.0), (second, 0.2)):
        directory = root / case_slug(case)
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(
            json.dumps(_summary(case, offset)), encoding="utf-8"
        )

    loaded = load_complete_suite(root / case_slug(first), first)
    assert loaded is not None
    aggregate = aggregate_completed([first, second], root)
    assert aggregate["completed_cases"] == 2
    assert aggregate["methods"][0]["mean_visible_J_and_F"] == 0.1

    corrupted = root / case_slug(first) / "summary.json"
    payload = json.loads(corrupted.read_text(encoding="utf-8"))
    payload["sanity_checks"]["last_mask_equals_replay_1"] = False
    corrupted.write_text(json.dumps(payload), encoding="utf-8")
    assert load_complete_suite(corrupted.parent, first) is None


def test_attach_and_aggregate_temporal_metrics(tmp_path: Path) -> None:
    case = _case("a", "india", 1, 10, "reappearance")
    suite = tmp_path / case_slug(case)
    suite.mkdir(parents=True)
    summary = _summary(case)
    (suite / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    for method in REQUIRED_METHODS:
        method_dir = suite / method
        method_dir.mkdir()
        scores = [0.8, 0.8, 0.8] if method == "full_replay" else [0.0, 0.8, 0.8]
        report = {
            "frames": [
                {
                    "frame": frame,
                    "ground_truth_present": True,
                    "J_and_F": score,
                }
                for frame, score in zip(range(11, 14), scores, strict=True)
            ]
        }
        (method_dir / "davis.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

    upgraded = attach_temporal_metrics(summary, suite)
    assert upgraded["schema_version"] == "cmmt.cached_baseline_suite.v2"
    (suite / "summary.json").write_text(json.dumps(upgraded), encoding="utf-8")
    aggregate = aggregate_completed([case], tmp_path)
    temporal = aggregate["methods"][0]["temporal"]
    assert temporal["checkpoint_scores"]["1"]["mean_candidate_J_and_F_all"] == 0.0
    assert (
        temporal["checkpoint_scores"]["1"]["mean_candidate_J_and_F_visible"]
        == 0.0
    )
    assert temporal["mean_switch_shock_first_5_visible"] == pytest.approx(0.8 / 3)
