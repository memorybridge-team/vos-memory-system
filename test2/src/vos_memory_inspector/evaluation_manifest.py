"""Deterministic DAVIS video/object/switch manifests for CMMT evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


SCHEMA_VERSION = "cmmt.davis_evaluation_manifest.v1"
_BACKGROUND_LABEL = 0
_VOID_LABEL = 255


def _read_label_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image).copy()


def _read_split(root: Path, split: str) -> list[str]:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    split_file = root / "ImageSets" / "2017" / f"{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"DAVIS split file not found: {split_file}")
    sequences = [
        line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not sequences:
        raise ValueError(f"DAVIS split is empty: {split_file}")
    return sequences


def _mask_statistics(mask: np.ndarray, object_id: int) -> tuple[int, tuple[float, float] | None]:
    selected = mask == object_id
    area = int(selected.sum())
    if area == 0:
        return 0, None
    ys, xs = np.nonzero(selected)
    return area, (float(xs.mean()), float(ys.mean()))


def _transition_diagnostics(
    areas: list[int],
    centroids: list[tuple[float, float] | None],
    frame: int,
    *,
    diagonal: float,
) -> dict[str, float | int | None]:
    area = areas[frame]
    next_area = areas[frame + 1]
    ratio = None if area == 0 else float(next_area / area)
    inverse_ratio = None if next_area == 0 else float(area / next_area)
    centroid = centroids[frame]
    next_centroid = centroids[frame + 1]
    motion = None
    if centroid is not None and next_centroid is not None:
        motion = float(
            math.hypot(next_centroid[0] - centroid[0], next_centroid[1] - centroid[1])
            / diagonal
        )
    return {
        "area": area,
        "next_area": next_area,
        "next_over_current_area": ratio,
        "current_over_next_area": inverse_ratio,
        "centroid_displacement_over_diagonal": motion,
    }


def _regular_frames(eligible: list[int], quantiles: Iterable[float]) -> dict[int, list[str]]:
    selected: dict[int, list[str]] = defaultdict(list)
    for quantile in quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("regular quantiles must be in [0, 1]")
        index = int(round(quantile * (len(eligible) - 1)))
        selected[eligible[index]].append(f"regular_q{int(round(quantile * 100)):02d}")
    return selected


def _event_frames(
    eligible: list[int],
    areas: list[int],
    centroids: list[tuple[float, float] | None],
    *,
    diagonal: float,
    area_drop_ratio: float,
    area_growth_ratio: float,
) -> dict[int, list[str]]:
    selected: dict[int, list[str]] = defaultdict(list)

    disappearance = [frame for frame in eligible if areas[frame] > 0 and areas[frame + 1] == 0]
    if disappearance:
        selected[disappearance[0]].append("full_occlusion_entry")
    else:
        drops = [
            (areas[frame + 1] / areas[frame], frame)
            for frame in eligible
            if areas[frame] > 0
        ]
        if drops and min(drops)[0] <= area_drop_ratio:
            selected[min(drops)[1]].append("strong_area_drop")

    reappearance = [frame for frame in eligible if areas[frame] == 0 and areas[frame + 1] > 0]
    if reappearance:
        selected[reappearance[0]].append("reappearance")
    else:
        growth = [
            (areas[frame + 1] / areas[frame], frame)
            for frame in eligible
            if areas[frame] > 0
        ]
        if growth and max(growth)[0] >= area_growth_ratio:
            selected[max(growth)[1]].append("strong_area_growth")

    motion = []
    for frame in eligible:
        current = centroids[frame]
        following = centroids[frame + 1]
        if current is None or following is None:
            continue
        normalized = math.hypot(following[0] - current[0], following[1] - current[1]) / diagonal
        motion.append((normalized, frame))
    if motion:
        selected[max(motion)[1]].append("max_centroid_motion")
    return selected


def build_davis_evaluation_manifest(
    root: str | Path,
    *,
    split: str = "val",
    resolution: str = "480p",
    regular_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    min_prefix_frames: int = 5,
    min_future_frames: int = 20,
    area_drop_ratio: float = 0.35,
    area_growth_ratio: float = 3.0,
    seed: int = 7,
) -> dict[str, Any]:
    """Build a path-independent manifest from DAVIS ground-truth event metadata.

    Ground truth is used only to lock evaluation cases and diagnostic slice tags.
    The resulting manifest never contains pixels and must not be used as model input.
    """

    root = Path(root).resolve()
    if min_prefix_frames < 1 or min_future_frames < 1:
        raise ValueError("minimum prefix and future lengths must be positive")
    if not 0.0 < area_drop_ratio < 1.0:
        raise ValueError("area_drop_ratio must be in (0, 1)")
    if area_growth_ratio <= 1.0:
        raise ValueError("area_growth_ratio must be greater than 1")

    cases: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    sequences = _read_split(root, split)
    for sequence in sequences:
        frame_dir = root / "JPEGImages" / resolution / sequence
        annotation_dir = root / "Annotations" / resolution / sequence
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        annotation_paths = sorted(annotation_dir.glob("*.png"))
        if not frame_paths or len(annotation_paths) != len(frame_paths):
            raise ValueError(
                f"DAVIS sequence {sequence!r} needs one annotation per JPG; "
                f"frames={len(frame_paths)}, annotations={len(annotation_paths)}"
            )
        expected_stems = [f"{index:05d}" for index in range(len(frame_paths))]
        if [path.stem for path in frame_paths] != expected_stems:
            raise ValueError(f"non-contiguous JPG frame numbering in {sequence!r}")
        if [path.stem for path in annotation_paths] != expected_stems:
            raise ValueError(f"non-contiguous annotation numbering in {sequence!r}")

        first_mask = _read_label_image(annotation_paths[0])
        object_ids = sorted(
            int(value)
            for value in np.unique(first_mask)
            if int(value) not in {_BACKGROUND_LABEL, _VOID_LABEL}
        )
        if not object_ids:
            raise ValueError(f"first annotation has no objects: {annotation_paths[0]}")
        masks = [_read_label_image(path) for path in annotation_paths]
        if any(mask.shape != first_mask.shape for mask in masks):
            raise ValueError(f"annotation shape changes within {sequence!r}")
        height, width = first_mask.shape
        diagonal = math.hypot(width, height)
        frame_count = len(frame_paths)
        evaluation_last_frame = frame_count - 2
        first_switch = min_prefix_frames - 1
        last_switch = evaluation_last_frame - min_future_frames
        eligible = list(range(first_switch, last_switch + 1))

        for object_id in object_ids:
            if not eligible:
                excluded.append(
                    {
                        "sequence": sequence,
                        "object_id": object_id,
                        "reason": "insufficient_prefix_or_future_frames",
                        "frame_count": frame_count,
                    }
                )
                continue
            stats = [_mask_statistics(mask, object_id) for mask in masks]
            areas = [item[0] for item in stats]
            centroids = [item[1] for item in stats]
            selected = _regular_frames(eligible, regular_quantiles)
            for frame, tags in _event_frames(
                eligible,
                areas,
                centroids,
                diagonal=diagonal,
                area_drop_ratio=area_drop_ratio,
                area_growth_ratio=area_growth_ratio,
            ).items():
                selected[frame].extend(tags)
            for switch_frame in sorted(selected):
                cases.append(
                    {
                        "case_id": f"{split}:{sequence}:obj{object_id}:switch{switch_frame}",
                        "split": split,
                        "sequence": sequence,
                        "object_id": object_id,
                        "switch_frame": switch_frame,
                        "future_end_frame": evaluation_last_frame,
                        "frame_count": frame_count,
                        "tags": sorted(set(selected[switch_frame])),
                        "transition": _transition_diagnostics(
                            areas, centroids, switch_frame, diagonal=diagonal
                        ),
                    }
                )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "DAVIS 2017 trainval 480p",
        "dataset_root_policy": "runtime_argument_not_stored",
        "ground_truth_policy": (
            "GT selects and tags cases only; GT pixels are never model inputs."
        ),
        "split": split,
        "resolution": resolution,
        "seed": seed,
        "selection_policy": {
            "regular_quantiles": list(regular_quantiles),
            "min_prefix_frames": min_prefix_frames,
            "min_future_frames": min_future_frames,
            "area_drop_ratio": area_drop_ratio,
            "area_growth_ratio": area_growth_ratio,
            "last_video_frame_excluded": True,
        },
        "sequence_count": len(sequences),
        "case_count": len(cases),
        "cases": cases,
        "excluded": excluded,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return manifest


def write_evaluation_manifest(manifest: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported evaluation manifest: {manifest.get('schema_version')!r}")
    digest = manifest.pop("content_sha256", None)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest["content_sha256"] = digest
    if digest != expected:
        raise ValueError("evaluation manifest content_sha256 does not match its content")
    return manifest
