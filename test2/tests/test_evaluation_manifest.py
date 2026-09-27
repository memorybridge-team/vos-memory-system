from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vos_memory_inspector.evaluation_manifest import (
    build_davis_evaluation_manifest,
    load_evaluation_manifest,
    write_evaluation_manifest,
)


def _synthetic_davis(root: Path) -> None:
    frames = root / "JPEGImages" / "480p" / "event-demo"
    masks = root / "Annotations" / "480p" / "event-demo"
    split = root / "ImageSets" / "2017"
    frames.mkdir(parents=True)
    masks.mkdir(parents=True)
    split.mkdir(parents=True)
    split.joinpath("val.txt").write_text("event-demo\n", encoding="utf-8")

    for frame in range(12):
        frames.joinpath(f"{frame:05d}.jpg").write_bytes(b"frame-placeholder")
        mask = np.zeros((12, 16), dtype=np.uint8)
        if frame <= 3:
            mask[3:6, 1:4] = 1
        elif frame >= 6:
            x = 10 if frame == 6 else 2
            mask[3:6, x : x + 3] = 1
        Image.fromarray(mask).save(masks / f"{frame:05d}.png")


def test_manifest_is_deterministic_and_tags_events(tmp_path: Path) -> None:
    root = tmp_path / "DAVIS"
    _synthetic_davis(root)

    first = build_davis_evaluation_manifest(
        root,
        split="val",
        regular_quantiles=(0.25, 0.5, 0.75),
        min_prefix_frames=2,
        min_future_frames=3,
    )
    second = build_davis_evaluation_manifest(
        root,
        split="val",
        regular_quantiles=(0.25, 0.5, 0.75),
        min_prefix_frames=2,
        min_future_frames=3,
    )

    assert first == second
    assert first["sequence_count"] == 1
    assert first["case_count"] >= 3
    tags = {tag for case in first["cases"] for tag in case["tags"]}
    assert "full_occlusion_entry" in tags
    assert "reappearance" in tags
    assert "max_centroid_motion" in tags
    assert all("dataset_root" not in case for case in first["cases"])

    output = tmp_path / "manifest.json"
    write_evaluation_manifest(first, output)
    assert load_evaluation_manifest(output) == first


def test_manifest_checksum_detects_manual_change(tmp_path: Path) -> None:
    root = tmp_path / "DAVIS"
    _synthetic_davis(root)
    manifest = build_davis_evaluation_manifest(
        root,
        min_prefix_frames=2,
        min_future_frames=3,
    )
    output = tmp_path / "manifest.json"
    write_evaluation_manifest(manifest, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["seed"] = 99
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="content_sha256"):
        load_evaluation_manifest(output)


def test_manifest_excludes_sequences_without_required_horizon(tmp_path: Path) -> None:
    root = tmp_path / "DAVIS"
    _synthetic_davis(root)
    manifest = build_davis_evaluation_manifest(
        root,
        min_prefix_frames=8,
        min_future_frames=8,
    )

    assert manifest["case_count"] == 0
    assert manifest["excluded"][0]["reason"] == "insufficient_prefix_or_future_frames"
