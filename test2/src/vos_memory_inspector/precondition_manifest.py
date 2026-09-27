"""Leakage-safe LVOS v2/VOST train manifests for state-pair extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "cmmt.precondition_video_split.v1"


def _content_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _find_layout(root: Path, split: str) -> tuple[Path, Path, Path]:
    candidates = (
        (root / split / "JPEGImages", root / split / "Annotations"),
        (root / "JPEGImages" / split, root / "Annotations" / split),
        (root / "JPEGImages", root / "Annotations"),
    )
    for image_dir, annotation_dir in candidates:
        if image_dir.is_dir() and annotation_dir.is_dir():
            return image_dir, annotation_dir, image_dir.parent
    raise FileNotFoundError(
        f"could not find JPEGImages/Annotations for split {split!r} under {root}"
    )


def _metadata_groups(root: Path, split_root: Path, split: str) -> tuple[dict[str, str], dict[str, Any] | None]:
    candidates = (
        split_root / f"{split}_meta.json",
        root / "metadata" / f"{split}_meta.json",
        root / "metadata" / ("valid_meta.json" if split == "val" else f"{split}_meta.json"),
        split_root / "meta.json",
        split_root / "metadata.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result: dict[str, str] = {}
        for video_id, video in payload.get("videos", {}).items():
            group = (
                video.get("source_video_id")
                or video.get("source_video")
                or video.get("original_video_id")
                or video_id
            )
            result[str(video_id)] = str(group)
        return result, {
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {}, None


def _frame_ids(video_dir: Path) -> list[str]:
    extensions = {".jpg", ".jpeg", ".png"}
    values = sorted(path.stem for path in video_dir.iterdir() if path.suffix.lower() in extensions)
    if not values:
        raise ValueError(f"video has no frames: {video_dir}")
    return values


def _assign_groups(groups: list[str], seed: int) -> dict[str, str]:
    ranked = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )
    count = len(ranked)
    if count < 3:
        raise ValueError("at least three independent video groups are required")
    # Nominal 80/10/10 boundaries, clamped so that fit, dev and test each keep at
    # least one group. For count >= 10 the clamps are inactive.
    fit_end = min(max(1, int(count * 0.8)), count - 2)
    dev_end = min(max(fit_end + 1, int(count * 0.9)), count - 1)
    result = {}
    for index, group in enumerate(ranked):
        result[group] = "fit" if index < fit_end else "dev" if index < dev_end else "test"
    return result


def build_train_split_manifest(
    root: str | Path,
    *,
    dataset: str,
    release: str,
    official_split: str = "train",
    seed: int = 7,
    screening_counts: tuple[int, int, int] = (16, 2, 2),
) -> dict[str, Any]:
    """Freeze 80/10/10 groups before any model state is extracted."""

    root = Path(root).resolve()
    images, annotations, split_root = _find_layout(root, official_split)
    video_ids = sorted(
        path.name for path in images.iterdir() if path.is_dir() and (annotations / path.name).is_dir()
    )
    if not video_ids:
        raise ValueError(f"no paired image/annotation videos found under {root}")
    metadata_groups, metadata_provenance = _metadata_groups(root, split_root, official_split)
    group_by_video = {video: metadata_groups.get(video, video) for video in video_ids}
    assignment = _assign_groups(sorted(set(group_by_video.values())), seed)
    videos = []
    for video_id in video_ids:
        frames = _frame_ids(images / video_id)
        annotation_ids = set(_frame_ids(annotations / video_id))
        annotated_frames = [frame for frame in frames if frame in annotation_ids]
        if not annotated_frames:
            raise ValueError(f"video has no frame-aligned annotations: {video_id}")
        videos.append(
            {
                "video_id": video_id,
                "source_group_id": group_by_video[video_id],
                "split": assignment[group_by_video[video_id]],
                "video_dir": str((images / video_id).relative_to(root)),
                "annotation_dir": str((annotations / video_id).relative_to(root)),
                "frame_ids": frames,
                "annotated_frame_ids": annotated_frames,
                "frame_id_to_predictor_idx": {frame: index for index, frame in enumerate(frames)},
            }
        )
    selected: dict[str, list[str]] = {}
    for split, requested in zip(("fit", "dev", "test"), screening_counts, strict=True):
        eligible = sorted(row["video_id"] for row in videos if row["split"] == split)
        if len(eligible) < requested:
            raise ValueError(
                f"{dataset} {split} contains {len(eligible)} videos, fewer than screening request {requested}"
            )
        selected[split] = eligible[:requested]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "release": release,
        "official_split": official_split,
        "dataset_root_policy": "runtime_argument_not_stored",
        "seed": seed,
        "split_policy": "source-group-level deterministic hash 80/10/10",
        "screening": {
            "fit_videos": screening_counts[0],
            "dev_videos": screening_counts[1],
            "test_videos": screening_counts[2],
            "selected": selected,
            "tiny_overfit": {
                "video_id": selected["fit"][0],
                "selection": "first fixed screening fit video",
            },
        },
        "sampling": {
            "switch_quantiles": [0.25, 0.5, 0.75],
            "max_records_per_object": 16,
            "max_records_per_video": 128,
        },
        "metadata": metadata_provenance,
        "videos": videos,
    }
    split_groups = {
        split: {row["source_group_id"] for row in videos if row["split"] == split}
        for split in ("fit", "dev", "test")
    }
    if any(
        split_groups[left] & split_groups[right]
        for left, right in (("fit", "dev"), ("fit", "test"), ("dev", "test"))
    ):
        raise AssertionError("source groups overlap across fit/dev/test")
    manifest["counts"] = {
        split: sum(row["split"] == split for row in videos)
        for split in ("fit", "dev", "test")
    }
    manifest["content_sha256"] = _content_hash(manifest)
    return manifest


def validate_train_split_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported train split manifest")
    content = dict(manifest)
    expected = content.pop("content_sha256", None)
    if expected != _content_hash(content):
        raise ValueError("train split manifest content SHA-256 mismatch")
    videos = manifest.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("manifest has no videos")
    ids = [str(row["video_id"]) for row in videos]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate video IDs")
    groups: dict[str, set[str]] = {"fit": set(), "dev": set(), "test": set()}
    for row in videos:
        split = str(row["split"])
        if split not in groups:
            raise ValueError(f"invalid internal split: {split}")
        groups[split].add(str(row["source_group_id"]))
        mapping = row.get("frame_id_to_predictor_idx", {})
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError(f"non-contiguous frame mapping for {row['video_id']}")
    if groups["fit"] & groups["dev"] or groups["fit"] & groups["test"] or groups["dev"] & groups["test"]:
        raise ValueError("source groups leak across internal splits")


def write_train_split_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    validate_train_split_manifest(manifest)
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    partial.replace(path)
