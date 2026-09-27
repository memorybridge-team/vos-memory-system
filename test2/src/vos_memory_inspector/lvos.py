"""LVOS v2 metadata-driven evaluation manifest support."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "cmmt.lvosv2_evaluation_manifest.v1"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("videos"), dict):
        raise ValueError(f"invalid LVOS metadata: {path}")
    return payload


def _split_root(root: Path, split: str) -> Path:
    candidate = root / split
    return candidate if candidate.is_dir() else root


def build_lvosv2_evaluation_manifest(
    root: str | Path,
    *,
    split: str = "val",
    regular_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    min_prefix_frames: int = 5,
    min_future_frames: int = 20,
    seed: int = 7,
) -> dict[str, Any]:
    """Build LVOS v2 cases from the official split metadata.

    LVOS metadata records object-specific frame ranges and attributes.  Switch
    candidates are selected only inside the object's annotated range, so a case
    cannot silently use frames before an object exists.  ``switch_frame`` is
    the original LVOS frame id, not an inferred zero-based position.
    """

    if min_prefix_frames < 1 or min_future_frames < 1:
        raise ValueError("minimum prefix and future lengths must be positive")
    for quantile in regular_quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("regular quantiles must be in [0, 1]")

    root = Path(root).resolve()
    split_root = _split_root(root, split)
    metadata_path = split_root / f"{split}_meta.json"
    if not metadata_path.is_file() and split == "val":
        metadata_path = split_root / "eval_meta.json"
    metadata = _load_json(metadata_path)
    attribute_path = split_root / f"{split}_meta_attribute.json"
    if not attribute_path.is_file() and split == "val":
        attribute_path = split_root / "eval_meta_attribute.json"
    attributes = _load_json(attribute_path) if attribute_path.is_file() else {"videos": {}}

    cases: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for video_id in sorted(metadata["videos"]):
        video = metadata["videos"][video_id]
        attr_video = attributes.get("videos", {}).get(video_id, {})
        for object_id in sorted(video.get("objects", {}), key=str):
            frame_range = video["objects"][object_id].get("frame_range", {})
            start = int(frame_range.get("start", 0))
            end = int(frame_range.get("end", start - 1))
            frame_nums = int(frame_range.get("frame_nums", end - start + 1))
            first_switch = start + min_prefix_frames - 1
            last_switch = end - min_future_frames
            eligible = list(range(first_switch, last_switch + 1))
            if not eligible:
                excluded.append(
                    {
                        "video_id": video_id,
                        "object_id": str(object_id),
                        "reason": "insufficient_prefix_or_future_frames",
                        "start_frame": start,
                        "end_frame": end,
                    }
                )
                continue
            selected = sorted(
                {
                    eligible[int(round(q * (len(eligible) - 1)))]
                    for q in regular_quantiles
                }
            )
            tags = sorted(set(attr_video.get("attributes", [])))
            for switch_frame in selected:
                cases.append(
                    {
                        "case_id": f"{split}:{video_id}:obj{object_id}:switch{switch_frame}",
                        "dataset": "LVOS v2",
                        "release": "v2",
                        "official_split": split,
                        "video_id": video_id,
                        "object_id": str(object_id),
                        "switch_frame": switch_frame,
                        "first_prompt_frame": start,
                        "future_end_frame": end,
                        "frame_count_for_object": frame_nums,
                        "tags": tags,
                        "future_gt_available": split in {"train", "val"},
                        "annotation_policy": "dense_validation_annotations",
                    }
                )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "LVOS v2",
        "release": "v2",
        "dataset_root_policy": "runtime_argument_not_stored",
        "split": split,
        "seed": seed,
        "ground_truth_policy": (
            "Object frame ranges and attributes come from official metadata; future "
            "GT is locally evaluable only for train/validation splits."
        ),
        "selection_policy": {
            "regular_quantiles": list(regular_quantiles),
            "min_prefix_frames": min_prefix_frames,
            "min_future_frames": min_future_frames,
            "switch_frame_indexing": "official_lvos_frame_id",
        },
        "sequence_count": len(metadata["videos"]),
        "case_count": len(cases),
        "cases": cases,
        "excluded": excluded,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return manifest

