from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


IDENTITY_ELIGIBLE = {"maskmem_features", "maskmem_pos_enc", "obj_ptr"}
SUPPORTING_STATE = {"pred_masks", "object_score_logits"}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _base_name(tensor_name: str) -> str:
    return tensor_name.split("[", 1)[0]


def _index(rows: Iterable[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["video_id"],
            row["frame_idx"],
            str(row["object_id"]),
            row["cond_non_cond"],
            row["tensor_name"],
        )
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key}")
        result[key] = row
    return result


def compare_manifests(
    source_path: str | Path,
    target_path: str | Path,
) -> dict[str, Any]:
    source_rows = load_jsonl(source_path)
    target_rows = load_jsonl(target_path)
    source = _index(source_rows)
    target = _index(target_rows)
    source_models = sorted({row["model_id"] for row in source_rows})
    target_models = sorted({row["model_id"] for row in target_rows})
    comparisons: list[dict[str, Any]] = []
    for key in sorted(set(source) | set(target), key=str):
        left = source.get(key)
        right = target.get(key)
        tensor_name = key[-1]
        base_name = _base_name(tensor_name)
        row: dict[str, Any] = {
            "key": list(key),
            "tensor_name": tensor_name,
            "source_present": left is not None,
            "target_present": right is not None,
            "source_shape": None if left is None else left["shape"],
            "target_shape": None if right is None else right["shape"],
            "source_dtype": None if left is None else left["dtype"],
            "target_dtype": None if right is None else right["dtype"],
            "source_num_bytes": None if left is None else left["num_bytes"],
            "target_num_bytes": None if right is None else right["num_bytes"],
        }
        both = left is not None and right is not None
        row["shape_equal"] = both and left["shape"] == right["shape"]
        row["dtype_equal"] = both and left["dtype"] == right["dtype"]
        row["num_bytes_equal"] = both and left["num_bytes"] == right["num_bytes"]
        source_spatial = None if left is None else _spatial(left["shape"])
        target_spatial = None if right is None else _spatial(right["shape"])
        row["spatial_resolution_equal"] = (
            None
            if not both or source_spatial is None or target_spatial is None
            else source_spatial == target_spatial
        )
        row["channel_equal"] = both and _channel(left["shape"]) == _channel(
            right["shape"]
        )
        if not both:
            disposition = "missing_counterpart"
        elif base_name in SUPPORTING_STATE:
            disposition = "supporting_state_not_primary_translator_target"
        elif base_name.startswith("memory_attention."):
            disposition = "transient_target_assembled_input"
        elif row["shape_equal"] and base_name in IDENTITY_ELIGIBLE:
            disposition = "direct_copy_shape_candidate_semantics_unverified"
        elif base_name in IDENTITY_ELIGIBLE and _linear_axes_compatible(
            base_name, left["shape"], right["shape"]
        ):
            disposition = "linear_mapping_shape_candidate"
        else:
            disposition = "nonlinear_or_structural_adapter_candidate"
        row["disposition"] = disposition
        comparisons.append(row)

    summary_counts: dict[str, int] = defaultdict(int)
    for row in comparisons:
        summary_counts[row["disposition"]] += 1
    matched = [
        row for row in comparisons if row["source_present"] and row["target_present"]
    ]
    return {
        "source_manifest": str(Path(source_path).resolve()),
        "target_manifest": str(Path(target_path).resolve()),
        "source_models": source_models,
        "target_models": target_models,
        "matched_rows": len(matched),
        "all_matched_shapes_equal": bool(matched)
        and all(row["shape_equal"] for row in matched),
        "disposition_counts": dict(sorted(summary_counts.items())),
        "comparisons": comparisons,
    }


def write_compatibility_report(
    report: dict[str, Any],
    json_path: str | Path,
    markdown_path: str | Path | None = None,
) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if markdown_path is not None:
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_to_markdown(report), encoding="utf-8")


def _spatial(shape: list[int]) -> list[int] | None:
    return shape[-2:] if len(shape) >= 4 else None


def _channel(shape: list[int]) -> int | None:
    return shape[1] if len(shape) >= 2 else None


def _linear_axes_compatible(
    base_name: str, source_shape: list[int], target_shape: list[int]
) -> bool:
    if len(source_shape) != len(target_shape) or not source_shape:
        return False
    if source_shape[0] != target_shape[0]:
        return False
    if base_name == "obj_ptr":
        return len(source_shape) == 2
    source_spatial = _spatial(source_shape)
    target_spatial = _spatial(target_shape)
    return source_spatial is not None and source_spatial == target_spatial


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAM 2 memory compatibility report",
        "",
        f"- Source model(s): `{', '.join(report['source_models'])}`",
        f"- Target model(s): `{', '.join(report['target_models'])}`",
        f"- Matched rows: {report['matched_rows']}",
        f"- All matched shapes equal: {report['all_matched_shapes_equal']}",
        "",
        "| Tensor | Frame | Object | Source shape | Target shape | Dtype equal | Disposition |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in report["comparisons"]:
        key = row["key"]
        lines.append(
            "| {tensor} | {frame} | {obj} | `{left}` | `{right}` | {dtype} | {disp} |".format(
                tensor=row["tensor_name"],
                frame=key[1],
                obj=key[2],
                left=row["source_shape"],
                right=row["target_shape"],
                dtype=row["dtype_equal"],
                disp=row["disposition"],
            )
        )
    lines.extend(
        [
            "",
            "> Shape compatibility does not establish representational alignment. "
            "A direct-copy candidate still requires a future-frame rollout test.",
            "",
        ]
    )
    return "\n".join(lines)
