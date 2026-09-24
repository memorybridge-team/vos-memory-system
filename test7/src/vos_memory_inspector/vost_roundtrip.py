"""Small official-validation VOST Base+ identity handoff diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image

from .device import resolve_device
from .base_roundtrip import (
    CONFIG_FILE,
    EXPECTED_CHECKPOINT_SHA256,
    Clip,
    Prompt,
    run_video,
    sha256_file,
    write_svg,
)
from .upstream import verify_sam2_checkout


VIDEO_IDS = ("555_tear_aluminium_foil", "6922_split_paper")


def _frame_number(path: Path) -> int:
    match = re.fullmatch(r"frame(\d+)", path.stem)
    if match is None:
        raise ValueError(f"unexpected VOST frame name: {path.name}")
    return int(match.group(1))


def validate_downloads(data_root: Path) -> dict[str, Any]:
    manifest_path = data_root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "VOST validation":
        raise ValueError("unexpected VOST download manifest")
    if tuple(manifest.get("videos", ())) != VIDEO_IDS:
        raise ValueError("downloaded VOST sequence selection differs from the experiment")
    if manifest.get("split") != "val":
        raise ValueError("the VOST diagnostic must use the validation split")
    files = manifest.get("files", [])
    seen: set[str] = set()
    total_bytes = 0
    for entry in files:
        name = str(entry["filename"])
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in seen:
            raise ValueError(f"invalid or duplicate dataset path: {name}")
        seen.add(name)
        path = data_root / relative
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"wrong byte count for downloaded file: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for downloaded file: {path}")
        total_bytes += path.stat().st_size
    split_file = data_root / "ImageSets/val.txt"
    if tuple(split_file.read_text(encoding="utf-8").split()) != VIDEO_IDS:
        raise ValueError("VOST ImageSets/val.txt does not match the selected subset")
    return {
        "files": len(files),
        "bytes": total_bytes,
        "sha256_download_manifest": sha256_file(manifest_path),
        "videos": list(VIDEO_IDS),
    }


def load_clip(
    data_root: Path,
    video_id: str,
    *,
    sam2_frames_root: Path | None = None,
) -> Clip:
    """Load one complete VOST val sequence; annotations are available at 5 fps."""
    image_dir = data_root / "JPEGImages" / video_id
    annotation_dir = data_root / "Annotations" / video_id
    frames = tuple(sorted(image_dir.glob("*.jpg"), key=_frame_number))
    if not frames:
        raise FileNotFoundError(f"no VOST RGB frames found for {video_id}")
    frame_ids = tuple(_frame_number(path) for path in frames)
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError(f"{video_id}: duplicate frame IDs")
    annotations = tuple(annotation_dir / f"{path.stem}.png" for path in frames)
    if any(not path.is_file() for path in annotations):
        raise FileNotFoundError(f"{video_id}: an RGB frame has no paired VOST mask")
    if len(tuple(annotation_dir.glob("*.png"))) != len(frames):
        raise ValueError(f"{video_id}: RGB/mask counts differ")

    with Image.open(annotations[0]) as image:
        first_labels = np.asarray(image)
    if first_labels.ndim != 2:
        raise ValueError(f"{video_id}: expected single-channel instance labels")
    object_ids = tuple(
        int(value) for value in np.unique(first_labels) if int(value) not in (0, 255)
    )
    if not object_ids:
        raise ValueError(f"{video_id}: first-frame prompt contains no object")
    if max(object_ids) >= 255:
        raise ValueError(f"{video_id}: object ID 255 is reserved for void pixels")

    unseen_ids: set[int] = set()
    for annotation in annotations:
        with Image.open(annotation) as image:
            labels = np.asarray(image)
        unseen_ids.update(
            int(value) for value in np.unique(labels)
            if int(value) not in (0, 255) and int(value) not in object_ids
        )
    if unseen_ids:
        raise ValueError(
            f"{video_id}: labels appear later that are absent from the initial prompt: "
            f"{sorted(unseen_ids)}"
        )

    sam2_frames = frames
    if sam2_frames_root is not None:
        staged_dir = sam2_frames_root / video_id
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_paths: list[Path] = []
        expected_names: set[str] = set()
        for index, source in enumerate(frames):
            name = f"{index:08d}.jpg"
            expected_names.add(name)
            link = staged_dir / name
            source_path = source.resolve()
            if link.is_symlink():
                if link.resolve() != source_path:
                    raise FileExistsError(f"staging link points to a different image: {link}")
            elif link.exists():
                raise FileExistsError(f"refusing to replace a non-symlink staging frame: {link}")
            else:
                link.symlink_to(source_path)
            staged_paths.append(link)
        unexpected = {
            path.name for path in staged_dir.glob("*.jpg") if path.name not in expected_names
        }
        if unexpected:
            raise ValueError(f"{video_id}: unexpected staged SAM 2 frames: {sorted(unexpected)}")
        sam2_frames = tuple(staged_paths)

    switch_index = len(frames) // 2
    if switch_index <= 0 or switch_index >= len(frames) - 1:
        raise ValueError(f"{video_id}: sequence is too short for a mid-sequence transfer")
    prompt = tuple(Prompt(object_id, 0, frame_ids[0]) for object_id in object_ids)
    return Clip(
        video_id=video_id,
        frames=sam2_frames,
        annotations=annotations,
        original_frame_ids=frame_ids,
        prompts=prompt,
        switch_index=switch_index,
    )


def _load_official_vost_evaluator(repository: Path):
    evaluation_dir = repository.resolve() / "evaluation"
    if not (evaluation_dir / "source/evaluation.py").is_file():
        raise FileNotFoundError(f"official VOST evaluator is missing: {evaluation_dir}")
    if str(evaluation_dir) not in sys.path:
        sys.path.insert(0, str(evaluation_dir))
    # VOST's 2023 evaluator uses the removed np.bool alias; restore only that
    # compatibility name in this process without editing the official source.
    if not hasattr(np, "bool"):
        np.bool = np.bool_
    try:
        from source.evaluation import Evaluation
        from source.metrics import db_eval_boundary, db_eval_iou
    except ImportError as exc:
        raise RuntimeError(
            "official VOST metrics require scipy, opencv-python, scikit-image, and tqdm"
        ) from exc
    commit = __import__("subprocess").run(
        ["git", "-C", str(repository.resolve()), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return Evaluation, db_eval_iou, db_eval_boundary, commit


def _summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric_name in ("J", "J_last"):
        values = metrics[metric_name]
        result[metric_name] = {
            "mean": float(np.mean(values["M"])),
            "recall": float(np.mean(values["R"])),
            "decay": float(np.mean(values["D"])),
            "per_object_mean": {
                key: float(value) for key, value in values["M_per_object"].items()
            },
        }
    return result


def run_experiment(
    *,
    data_root: Path,
    sam2_repo: Path,
    checkpoint: Path,
    evaluator_repo: Path,
    output_root: Path,
    device: str = "auto",
    seed: int = 7,
) -> dict[str, Any]:
    download_validation = validate_downloads(data_root)
    actual_device = resolve_device(device)
    sam2_commit = verify_sam2_checkout(sam2_repo)
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"unexpected Base+ checkpoint SHA-256: {checkpoint_hash}")
    Evaluation, db_eval_iou, db_eval_boundary, evaluator_commit = (
        _load_official_vost_evaluator(evaluator_repo)
    )

    output_root.mkdir(parents=True, exist_ok=False)
    prediction_roots = {
        "native": output_root / "predictions/native",
        "transferred": output_root / "predictions/transferred",
    }
    clips = [
        load_clip(
            data_root, video_id, sam2_frames_root=output_root / "sam2_input_frames"
        )
        for video_id in VIDEO_IDS
    ]
    summaries: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    # Match the official VOST evaluator: 255 is converted to background, not ignored.
    def official_iou(annotation, prediction, void_pixels=None):
        return db_eval_iou(annotation, prediction, void_pixels=None)

    def official_boundary(annotation, prediction, void_pixels=None):
        return db_eval_boundary(annotation, prediction, void_pixels=None)

    for clip in clips:
        rows, summary = run_video(
            clip,
            sam2_repo=sam2_repo,
            checkpoint=checkpoint,
            device=actual_device,
            iou_metric=official_iou,
            boundary_metric=official_boundary,
            seed=seed,
            prediction_roots=prediction_roots,
        )
        video_dir = output_root / clip.video_id
        video_dir.mkdir(parents=True, exist_ok=False)
        with (video_dir / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_svg(clip, rows, video_dir / "jf_curve.svg")
        (video_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        summaries.append(summary)
        metric_rows.extend(rows)

    official_evaluation = Evaluation(
        dataset_root=str(data_root), gt_set="val", sequences=list(VIDEO_IDS)
    )
    official_native = _summary_metrics(official_evaluation.evaluate(str(prediction_roots["native"])))
    official_transferred = _summary_metrics(
        official_evaluation.evaluate(str(prediction_roots["transferred"]))
    )

    post_rows = [
        row for row in metric_rows
        if row["phase"] == "post_switch" and row["included_in_mean"]
    ]
    report = {
        "schema_version": "cmmt.vost_base_roundtrip.v1",
        "status": "passed" if all(item["passed"] for item in summaries) else "failed",
        "claim": (
            "VOST two-sequence same-Base+ clean-Target identity handoff diagnostic; "
            "not a trained cross-model transfer result"
        ),
        "selection": "two complete VOST validation sequences, 36 MB total RGB+mask payload",
        "prompt_policy": "all object IDs in each sequence's first annotated frame; first frame only",
        "switch_policy": "per-sequence midpoint; Target continuation starts at switch index + 1",
        "scoring_policy": (
            "per-frame J/F uses the official VOST metric functions; supplemental curves include "
            "all frames, while the official aggregate J/J_last evaluator excludes first and last frames"
        ),
        "official_vost_metrics": {
            "native": official_native,
            "transferred": official_transferred,
            "delta_J_mean": official_transferred["J"]["mean"] - official_native["J"]["mean"],
            "delta_J_last_mean": (
                official_transferred["J_last"]["mean"] - official_native["J_last"]["mean"]
            ),
        },
        "supplemental_post_switch_JF": {
            "rows": len(post_rows),
            "native_mean": mean(row["native_JF"] for row in post_rows),
            "transferred_mean": mean(row["transfer_JF"] for row in post_rows),
            "mean_delta": mean(row["delta_JF"] for row in post_rows),
            "max_abs_delta": max(abs(row["delta_JF"]) for row in post_rows),
        },
        "dataset": download_validation,
        "videos": VIDEO_IDS,
        "device": actual_device,
        "seed": seed,
        "sam2_commit": sam2_commit,
        "sam2_checkpoint_sha256": checkpoint_hash,
        "sam2_config": CONFIG_FILE,
        "vost_evaluator_commit": evaluator_commit,
        "results": summaries,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
