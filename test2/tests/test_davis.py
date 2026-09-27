from __future__ import annotations

from zipfile import ZipFile

import pytest

from vos_memory_inspector.davis import _safe_extract, validate_davis_sequence


def test_validate_davis_val_sequence_resolves_first_prompt(tmp_path) -> None:
    root = tmp_path / "DAVIS"
    frames = root / "JPEGImages" / "480p" / "demo"
    masks = root / "Annotations" / "480p" / "demo"
    split = root / "ImageSets" / "2017"
    frames.mkdir(parents=True)
    masks.mkdir(parents=True)
    split.mkdir(parents=True)
    (frames / "00000.jpg").write_bytes(b"synthetic")
    (frames / "00001.jpg").write_bytes(b"synthetic")
    (masks / "00000.png").write_bytes(b"synthetic")
    (split / "val.txt").write_text("demo\n", encoding="utf-8")
    sequence = validate_davis_sequence(root, "demo")
    assert sequence.frame_count == 2
    assert sequence.first_mask == masks / "00000.png"


def test_validate_davis_train_sequence(tmp_path) -> None:
    root = tmp_path / "DAVIS"
    frames = root / "JPEGImages" / "480p" / "train-demo"
    masks = root / "Annotations" / "480p" / "train-demo"
    split = root / "ImageSets" / "2017"
    frames.mkdir(parents=True)
    masks.mkdir(parents=True)
    split.mkdir(parents=True)
    (frames / "00000.jpg").write_bytes(b"synthetic")
    (masks / "00000.png").write_bytes(b"synthetic")
    (split / "train.txt").write_text("train-demo\n", encoding="utf-8")

    sequence = validate_davis_sequence(root, "train-demo", split="train")

    assert sequence.name == "train-demo"


def test_validate_davis_rejects_unknown_split(tmp_path) -> None:
    root = tmp_path / "DAVIS"
    frames = root / "JPEGImages" / "480p" / "demo"
    masks = root / "Annotations" / "480p" / "demo"
    frames.mkdir(parents=True)
    masks.mkdir(parents=True)

    with pytest.raises(ValueError, match="split must be"):
        validate_davis_sequence(root, "demo", split="test")


def test_safe_extract_rejects_parent_traversal(tmp_path) -> None:
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("../outside.txt", "not extracted")
    with ZipFile(archive) as zip_file, pytest.raises(ValueError, match="Unsafe path"):
        _safe_extract(zip_file, tmp_path / "destination")

