from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from vos_memory_inspector.artifacts import sam2_frame_paths, write_handoff_artifacts


def test_sam2_frame_paths_follow_predictor_order(tmp_path) -> None:
    for name in ("10.jpg", "9.jpg", "100.jpeg", "11.png"):
        Image.new("RGB", (2, 2)).save(tmp_path / name)
    # SAM 2 reads JPEG only and sorts by int(stem), not lexically.
    assert [path.name for path in sam2_frame_paths(tmp_path)] == ["9.jpg", "10.jpg", "100.jpeg"]
    (tmp_path / "frame.jpg").write_bytes(b"")
    with pytest.raises(ValueError, match="integer frame file names"):
        sam2_frame_paths(tmp_path)


def test_write_handoff_artifacts(tmp_path) -> None:
    video_dir = tmp_path / "frames"
    video_dir.mkdir()
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    image[..., 2] = 80
    # File numbering starts at 5; predictor index 2 is therefore 00015.jpg.
    for stem in ("00005", "00010", "00015"):
        Image.fromarray(image).save(video_dir / f"{stem}.jpg")
    # Non-JPEG files are not SAM 2 frames and must not shift indices.
    Image.fromarray(image).save(video_dir / "00007.png")
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    ground_truth = np.zeros((12, 20), dtype=np.uint8)
    ground_truth[1:4, 2:5] = 1
    Image.fromarray(ground_truth).save(annotation_dir / "00015.png")
    oracle = torch.full((1, 1, 12, 20), -1.0)
    oracle[..., 2:8, 4:10] = 1.0
    candidate = oracle.clone()
    candidate[..., 6:10, 8:14] = 1.0
    output_dir = tmp_path / "artifacts"
    manifest = write_handoff_artifacts(
        video_dir=video_dir,
        annotation_dir=annotation_dir,
        object_id=1,
        oracle_masks={2: oracle},
        candidate_masks={2: candidate},
        output_dir=output_dir,
        report={
            "source_model_id": "tiny",
            "target_model_id": "large",
            "translator": "direct_copy",
            "switch_frame": 1,
            "mask_comparison_to_target_native": {
                "mean_binary_iou": 0.5,
                "mean_mse": 1.0,
            },
        },
        candidate_label="Candidate",
    )
    assert (output_dir / "oracle_masks/00002.png").is_file()
    assert (output_dir / "candidate_masks/00002.png").is_file()
    assert (output_dir / manifest["comparisons"][0]).is_file()
    assert Image.open(output_dir / manifest["comparisons"][0]).size == (80, 40)
    saved_report = json.loads((output_dir / "report.json").read_text())
    assert saved_report["artifacts"]["comparisons"] == [
        "comparisons/frame_00002.png"
    ]
    assert not list(output_dir.rglob("*.partial"))


def test_write_handoff_artifacts_rejects_missing_predictor_frame(tmp_path) -> None:
    video_dir = tmp_path / "frames"
    video_dir.mkdir()
    Image.new("RGB", (4, 4)).save(video_dir / "00000.jpg")
    mask = torch.zeros((1, 1, 4, 4))
    with pytest.raises(FileNotFoundError, match="video frame 3"):
        write_handoff_artifacts(
            video_dir=video_dir,
            annotation_dir=tmp_path,
            object_id=1,
            oracle_masks={3: mask},
            candidate_masks={3: mask},
            output_dir=tmp_path / "out",
            report={},
            candidate_label="Candidate",
        )
