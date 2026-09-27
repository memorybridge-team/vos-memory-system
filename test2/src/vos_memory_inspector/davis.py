from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


DAVIS_2017_TRAINVAL_480P_URL = (
    "https://data.vision.ee.ethz.ch/csergi/share/davis/"
    "DAVIS-2017-trainval-480p.zip"
)


@dataclass(frozen=True)
class DavisSequence:
    name: str
    frames_directory: Path
    first_mask: Path
    frame_count: int
    resolution: str


def validate_davis_sequence(
    root: str | Path,
    sequence: str,
    *,
    resolution: str = "480p",
    split: str | None = "val",
) -> DavisSequence:
    root = Path(root).resolve()
    frames = root / "JPEGImages" / resolution / sequence
    annotations = root / "Annotations" / resolution / sequence
    if not frames.is_dir():
        raise FileNotFoundError(f"DAVIS frame directory not found: {frames}")
    if not annotations.is_dir():
        raise FileNotFoundError(f"DAVIS annotation directory not found: {annotations}")
    if split is not None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train', 'val', or None")
        split_file = root / "ImageSets" / "2017" / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(
                f"DAVIS 2017 {split} split file not found: {split_file}"
            )
        split_sequences = {
            line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if sequence not in split_sequences:
            raise ValueError(
                f"Sequence {sequence!r} is not in DAVIS 2017 {split}.txt"
            )
    images = sorted(frames.glob("*.jpg"))
    masks = sorted(annotations.glob("*.png"))
    if not images:
        raise ValueError(f"No JPG frames found in {frames}")
    if not masks:
        raise ValueError(f"No PNG annotations found in {annotations}")
    image_stems = {path.stem for path in images}
    mask_stems = {path.stem for path in masks}
    if images[0].stem not in mask_stems:
        raise ValueError(
            f"First frame {images[0].name} has no matching annotation in {annotations}"
        )
    if not mask_stems.issubset(image_stems):
        extras = sorted(mask_stems - image_stems)[:5]
        raise ValueError(f"Annotations without matching frames: {extras}")
    return DavisSequence(
        name=sequence,
        frames_directory=frames,
        first_mask=annotations / f"{images[0].stem}.png",
        frame_count=len(images),
        resolution=resolution,
    )


def download_davis_2017_trainval_480p(
    destination: str | Path,
    *,
    accept_dataset_terms: bool,
    keep_archive: bool = False,
) -> Path:
    """Download and safely extract the official DAVIS 2017 trainval archive."""

    if not accept_dataset_terms:
        raise ValueError(
            "Dataset terms must be reviewed and accepted before download; "
            "pass --accept-dataset-terms to confirm."
        )
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "DAVIS-2017-trainval-480p.zip"
    partial = archive.with_suffix(".zip.part")
    with urlopen(DAVIS_2017_TRAINVAL_480P_URL) as response, partial.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    partial.replace(archive)
    with ZipFile(archive) as zip_file:
        _safe_extract(zip_file, destination)
    if not keep_archive:
        archive.unlink()
    davis_root = destination / "DAVIS"
    if not (davis_root / "ImageSets" / "2017" / "val.txt").is_file():
        raise RuntimeError(
            f"Downloaded archive did not produce the expected DAVIS root: {davis_root}"
        )
    return davis_root


def _safe_extract(zip_file: ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in zip_file.infolist():
        target = (destination / member.filename).resolve()
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise ValueError(f"Unsafe path in DAVIS archive: {member.filename}") from exc
    zip_file.extractall(destination)
