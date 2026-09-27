"""MOSEv2 validation manifest support.

MOSEv2 validation publishes RGB frames and only the first-frame annotation.  It
must therefore not be passed through the DAVIS event-tagging code: future GT
visibility and reappearance tags are unavailable locally and belong to the
official evaluation path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCHEMA_VERSION = "cmmt.mosev2_evaluation_manifest.v1"
_BACKGROUND_LABEL = 0
_VOID_LABEL = 255


def _split_root(root: Path, split: str) -> Path:
    candidate = root / split
    return candidate if candidate.is_dir() else root


def _first_frame_object_ids(path: Path) -> list[int]:
    with Image.open(path) as image:
        labels = np.asarray(image)
    return sorted(
        int(value)
        for value in np.unique(labels)
        if int(value) not in {_BACKGROUND_LABEL, _VOID_LABEL}
    )


def build_mosev2_evaluation_manifest(
    root: str | Path,
    *,
    split: str = "valid",
    regular_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    min_prefix_frames: int = 5,
    min_future_frames: int = 20,
    seed: int = 7,
) -> dict[str, Any]:
    """Build a path-independent MOSEv2 validation manifest.

    The validation split contains only first-frame masks.  Cases are therefore
    fixed by deterministic temporal quantiles, while all future GT-dependent
    tags are explicitly marked unavailable rather than inferred from pixels.
    ``switch_frame`` is the zero-based position in sorted RGB-frame order.
    """

    if min_prefix_frames < 1 or min_future_frames < 1:
        raise ValueError("minimum prefix and future lengths must be positive")
    for quantile in regular_quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("regular quantiles must be in [0, 1]")

    root = Path(root).resolve()
    split_root = _split_root(root, split)
    frames_root = split_root / "JPEGImages"
    annotations_root = split_root / "Annotations"
    if not frames_root.is_dir() or not annotations_root.is_dir():
        raise FileNotFoundError(
            f"MOSEv2 split must contain JPEGImages and Annotations: {split_root}"
        )

    cases: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    videos = sorted(path.name for path in frames_root.iterdir() if path.is_dir())
    for video in videos:
        frame_paths = sorted(frames_root.joinpath(video).glob("*.jpg"))
        annotation_paths = sorted(annotations_root.joinpath(video).glob("*.png"))
        if not frame_paths or not annotation_paths:
            excluded.append({"video_id": video, "reason": "missing_frames_or_first_mask"})
            continue
        if len(annotation_paths) != 1:
            raise ValueError(
                f"MOSEv2 validation expects one first-frame mask for {video!r}; "
                f"found {len(annotation_paths)}"
            )
        object_ids = _first_frame_object_ids(annotation_paths[0])
        if not object_ids:
            excluded.append({"video_id": video, "reason": "first_mask_has_no_objects"})
            continue
        frame_count = len(frame_paths)
        first_switch = min_prefix_frames - 1
        last_switch = frame_count - 1 - min_future_frames
        eligible = list(range(first_switch, last_switch + 1))
        if not eligible:
            for object_id in object_ids:
                excluded.append(
                    {
                        "video_id": video,
                        "object_id": object_id,
                        "reason": "insufficient_prefix_or_future_frames",
                        "frame_count": frame_count,
                    }
                )
            continue
        selected = sorted(
            {
                eligible[int(round(quantile * (len(eligible) - 1)))]
                for quantile in regular_quantiles
            }
        )
        frame_stems = [path.stem for path in frame_paths]
        for object_id in object_ids:
            for switch_frame in selected:
                cases.append(
                    {
                        "case_id": f"{split}:{video}:obj{object_id}:switch{switch_frame}",
                        "dataset": "MOSEv2",
                        "release": "v2",
                        "official_split": split,
                        "video_id": video,
                        "object_id": object_id,
                        "switch_frame": switch_frame,
                        "switch_frame_stem": frame_stems[switch_frame],
                        "first_prompt_frame_stem": annotation_paths[0].stem,
                        "future_end_frame": frame_count - 1,
                        "frame_count": frame_count,
                        "tags": ["validation_first_frame_only"],
                        "future_gt_available": False,
                        "annotation_policy": "first_frame_only",
                    }
                )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "MOSEv2",
        "release": "v2",
        "dataset_root_policy": "runtime_argument_not_stored",
        "split": split,
        "seed": seed,
        "annotation_policy": "validation_first_frame_only",
        "ground_truth_policy": (
            "First-frame masks identify objects and prompts only; future GT is not "
            "available locally and must be scored by the official evaluator."
        ),
        "selection_policy": {
            "regular_quantiles": list(regular_quantiles),
            "min_prefix_frames": min_prefix_frames,
            "min_future_frames": min_future_frames,
            "switch_frame_indexing": "zero_based_sorted_rgb_order",
        },
        "sequence_count": len(videos),
        "case_count": len(cases),
        "cases": cases,
        "excluded": excluded,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return manifest


def write_mosev2_evaluation_manifest(manifest: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

