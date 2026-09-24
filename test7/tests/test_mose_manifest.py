from __future__ import annotations

import json

import numpy as np
from PIL import Image

from vos_memory_inspector.mose import build_mosev2_evaluation_manifest


def test_mose_validation_manifest_marks_future_gt_unavailable(tmp_path):
    frames = tmp_path / "valid" / "JPEGImages" / "demo"
    annotations = tmp_path / "valid" / "Annotations" / "demo"
    frames.mkdir(parents=True)
    annotations.mkdir(parents=True)
    for index in range(12):
        frames.joinpath(f"{index:05d}.jpg").write_bytes(b"frame")
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    Image.fromarray(mask).save(annotations / "00000.png")

    manifest = build_mosev2_evaluation_manifest(
        tmp_path, min_prefix_frames=2, min_future_frames=3
    )

    assert manifest["schema_version"] == "cmmt.mosev2_evaluation_manifest.v1"
    assert manifest["sequence_count"] == 1
    assert manifest["case_count"] == 3
    assert all(case["future_gt_available"] is False for case in manifest["cases"])
    assert all(case["tags"] == ["validation_first_frame_only"] for case in manifest["cases"])
    assert all("dataset_root" not in case for case in manifest["cases"])

