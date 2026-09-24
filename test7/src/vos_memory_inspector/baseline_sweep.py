"""Deterministic selection and resumable bookkeeping for baseline sweeps."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from .temporal_evaluation import evaluate_temporal_handoff


RARE_EVENT_TAGS = frozenset(
    {
        "full_occlusion_entry",
        "reappearance",
        "strong_area_drop",
        "strong_area_growth",
    }
)
REQUIRED_METHODS = (
    "direct_copy",
    "target_reset",
    "last_mask",
    "replay_1",
    "replay_2",
    "replay_4",
    "full_replay",
)


def case_slug(case: Mapping[str, Any]) -> str:
    return (
        f"{case['sequence']}_obj{int(case['object_id'])}_"
        f"switch{int(case['switch_frame'])}"
    )


def select_cases(
    manifest: Mapping[str, Any],
    *,
    tags: Iterable[str] = RARE_EVENT_TAGS,
    case_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Select a stable subset without interpreting GT pixels at runtime."""

    selected_tags = frozenset(tags)
    requested_ids = None if case_ids is None else frozenset(case_ids)
    cases = []
    seen: set[str] = set()
    for raw_case in manifest["cases"]:
        case = dict(raw_case)
        case_id = str(case["case_id"])
        if case_id in seen:
            raise ValueError(f"duplicate case id in manifest: {case_id}")
        seen.add(case_id)
        if requested_ids is not None:
            include = case_id in requested_ids
        else:
            include = bool(selected_tags.intersection(case["tags"]))
        if include:
            cases.append(case)
    if requested_ids is not None:
        missing = sorted(requested_ids - seen)
        if missing:
            raise ValueError(f"requested case ids are absent from manifest: {missing}")
    return cases


def selection_manifest(
    source_manifest: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    *,
    tags: Iterable[str],
    case_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    policy = (
        {"match": "explicit_case_ids", "case_ids": sorted(set(case_ids))}
        if case_ids is not None
        else {"match": "any_tag", "tags": sorted(set(tags))}
    )
    payload: dict[str, Any] = {
        "schema_version": "cmmt.baseline_sweep_selection.v1",
        "source_manifest_content_sha256": source_manifest["content_sha256"],
        "selection_policy": policy,
        "case_count": len(cases),
        "cases": [dict(case) for case in cases],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def write_json_atomic(payload: Mapping[str, Any], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(output)


def cache_checksum_matches(path: str | Path) -> bool:
    path = Path(path)
    checksum = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not checksum.is_file():
        return False
    tokens = checksum.read_text(encoding="ascii").split()
    if not tokens:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == tokens[0]


def load_complete_suite(
    suite_directory: str | Path, case: Mapping[str, Any]
) -> dict[str, Any] | None:
    summary_path = Path(suite_directory) / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    expected = {
        "sequence": str(case["sequence"]),
        "object_id": int(case["object_id"]),
        "switch_frame": int(case["switch_frame"]),
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        return None
    methods = tuple(row.get("method") for row in summary.get("methods", []))
    if methods != REQUIRED_METHODS:
        return None
    if not summary.get("sanity_checks", {}).get("last_mask_equals_replay_1"):
        return None
    return summary


def attach_temporal_metrics(
    summary: Mapping[str, Any], suite_directory: str | Path
) -> dict[str, Any]:
    """Backfill temporal metrics from per-method DAVIS reports without inference."""

    suite_directory = Path(suite_directory)
    reports: dict[str, dict[str, Any]] = {}
    for row in summary["methods"]:
        method = str(row["method"])
        report_path = suite_directory / method / "davis.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"DAVIS report not found: {report_path}")
        reports[method] = json.loads(report_path.read_text(encoding="utf-8"))
    if "full_replay" not in reports:
        raise ValueError("full_replay DAVIS report is required as temporal reference")

    upgraded = dict(summary)
    upgraded["schema_version"] = "cmmt.cached_baseline_suite.v2"
    upgraded_methods = []
    for raw_row in summary["methods"]:
        row = dict(raw_row)
        method = str(row["method"])
        row["temporal"] = evaluate_temporal_handoff(
            reports[method],
            reports["full_replay"],
            switch_frame=int(summary["switch_frame"]),
        )
        upgraded_methods.append(row)
    upgraded["methods"] = upgraded_methods
    return upgraded


def aggregate_completed(
    cases: Sequence[Mapping[str, Any]], suite_root: str | Path
) -> dict[str, Any]:
    suite_root = Path(suite_root)
    completed: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for case in cases:
        summary = load_complete_suite(suite_root / case_slug(case), case)
        if summary is not None:
            completed.append((case, summary))

    method_rows: list[dict[str, Any]] = []
    for method_id in REQUIRED_METHODS:
        rows = [
            next(row for row in summary["methods"] if row["method"] == method_id)
            for _case, summary in completed
        ]
        if not rows:
            continue
        visible_values = [
            float(row["mean_visible_J_and_F"])
            for row in rows
            if row.get("mean_visible_J_and_F") is not None
        ]
        method_rows.append(
            {
                "method": method_id,
                "label": rows[0]["label"],
                "cases": len(rows),
                "mean_J_and_F": mean(float(row["mean_J_and_F"]) for row in rows),
                "mean_visible_J_and_F": (
                    mean(visible_values) if visible_values else None
                ),
                "mean_past_backbone_calls": mean(
                    float(row["backbone_calls_before_or_at_switch"])
                    for row in rows
                ),
                "mean_wall_time_seconds": mean(
                    float(row["wall_time_seconds"]) for row in rows
                ),
                "temporal": _aggregate_temporal(rows),
            }
        )
    return {
        "schema_version": "cmmt.cached_baseline_sweep_aggregate.v1",
        "planned_cases": len(cases),
        "completed_cases": len(completed),
        "completed_case_ids": [str(case["case_id"]) for case, _summary in completed],
        "methods": method_rows,
    }


def _aggregate_temporal(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    temporal = [row.get("temporal") for row in rows]
    if not temporal or any(value is None for value in temporal):
        return None

    def available_mean(values: Iterable[Any]) -> float | None:
        numeric = [float(value) for value in values if value is not None]
        return mean(numeric) if numeric else None

    checkpoints = {}
    for offset in ("1", "5", "20"):
        rows_at_offset = [
            value["checkpoint_scores"][offset]
            for value in temporal
            if value["checkpoint_scores"][offset]["available"]
        ]
        visible_rows = [
            row for row in rows_at_offset if row["ground_truth_present"]
        ]
        checkpoints[offset] = {
            "cases_available": len(rows_at_offset),
            "ground_truth_visible_cases": len(visible_rows),
            "mean_candidate_J_and_F_all": available_mean(
                row["candidate_J_and_F"] for row in rows_at_offset
            ),
            "mean_reference_gap_all": available_mean(
                row["reference_gap"] for row in rows_at_offset
            ),
            "mean_candidate_J_and_F_visible": available_mean(
                row["candidate_J_and_F"] for row in visible_rows
            ),
            "mean_reference_gap_visible": available_mean(
                row["reference_gap"] for row in visible_rows
            ),
        }
    recovered = [
        value["recovery"]["frames_from_switch"]
        for value in temporal
        if value["recovery"]["frames_from_switch"] is not None
    ]
    return {
        "checkpoint_scores": checkpoints,
        "mean_switch_shock_first_5_visible": available_mean(
            value["switch_shock"]["windows"]["5"]["mean_reference_gap"]
            for value in temporal
        ),
        "identity_break_proxy_rate": mean(
            1.0 if value["identity_break_proxy"]["occurred"] else 0.0
            for value in temporal
        ),
        "recovery_rate": len(recovered) / len(temporal),
        "mean_recovery_frames_when_recovered": available_mean(recovered),
        "recovery_censored_cases": len(temporal) - len(recovered),
    }


def write_aggregate_reports(report: Mapping[str, Any], output_root: str | Path) -> None:
    output_root = Path(output_root)
    write_json_atomic(report, output_root / "aggregate.json")
    lines = [
        "# Cached baseline rare-event sweep",
        "",
        f"- Completed: {report['completed_cases']}/{report['planned_cases']} cases",
        "- Scores are unweighted case means; this is not a full DAVIS benchmark.",
        "",
        "| Method | Cases | J&F | Visible J&F | Past backbone | Wall s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["methods"]:
        visible = row["mean_visible_J_and_F"]
        visible_text = "n/a" if visible is None else f"{visible:.6f}"
        lines.append(
            "| {label} | {cases} | {mean_J_and_F:.6f} | {visible} | "
            "{mean_past_backbone_calls:.2f} | {mean_wall_time_seconds:.3f} |".format(
                visible=visible_text, **row
            )
        )
    if report["methods"] and report["methods"][0].get("temporal") is not None:
        lines.extend(
            [
                "",
                "## Post-switch temporal metrics",
                "",
                "| Method | +1 J&F | +5 J&F | +20 J&F | Shock first 5 visible | Identity-loss rate | Recovery rate | Mean recovery frames |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["methods"]:
            temporal = row["temporal"]

            def value_text(value: Any) -> str:
                return "n/a" if value is None else f"{float(value):.6f}"

            checkpoints = temporal["checkpoint_scores"]
            lines.append(
                "| {label} | {plus1} | {plus5} | {plus20} | {shock} | "
                "{identity:.3f} | {recovery:.3f} | {recovery_frames} |".format(
                    label=row["label"],
                    plus1=value_text(
                        checkpoints["1"]["mean_candidate_J_and_F_visible"]
                    ),
                    plus5=value_text(
                        checkpoints["5"]["mean_candidate_J_and_F_visible"]
                    ),
                    plus20=value_text(
                        checkpoints["20"]["mean_candidate_J_and_F_visible"]
                    ),
                    shock=value_text(temporal["mean_switch_shock_first_5_visible"]),
                    identity=temporal["identity_break_proxy_rate"],
                    recovery=temporal["recovery_rate"],
                    recovery_frames=value_text(
                        temporal["mean_recovery_frames_when_recovered"]
                    ),
                )
            )
    markdown = output_root / "aggregate.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
