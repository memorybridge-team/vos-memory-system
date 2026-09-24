"""Human-viewable artifacts for checkpoint-backed handoff experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw


_FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _indexed_frames(video_dir: Path) -> dict[int, Path]:
    frames: dict[int, Path] = {}
    for path in video_dir.iterdir():
        if path.suffix.lower() not in _FRAME_EXTENSIONS:
            continue
        try:
            frame = int(path.stem)
        except ValueError:
            continue
        frames[frame] = path
    return frames


def _binary_mask(mask: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    array = mask.detach().cpu().float().numpy().squeeze()
    if array.ndim != 2:
        raise ValueError(f"expected one-object 2D mask logits, got {array.shape}")
    binary = Image.fromarray((array > 0).astype(np.uint8) * 255)
    if binary.size != size:
        binary = binary.resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(binary) > 0


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.astype(np.float32).copy()
    result[mask] = result[mask] * 0.45 + np.asarray(color, dtype=np.float32) * 0.55
    return np.clip(result, 0, 255).astype(np.uint8)


def _panel(image: np.ndarray, title: str) -> Image.Image:
    body = Image.fromarray(image)
    panel = Image.new("RGB", (body.width, body.height + 28), "white")
    panel.paste(body, (0, 28))
    ImageDraw.Draw(panel).text((8, 8), title, fill="black")
    return panel


def _comparison_canvas(
    image: Image.Image,
    ground_truth: np.ndarray,
    oracle: np.ndarray,
    candidate: np.ndarray,
    *,
    candidate_label: str,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    ground_truth_overlay = _overlay(rgb, ground_truth, (255, 45, 35))
    oracle_overlay = _overlay(rgb, oracle, (30, 210, 80))
    candidate_overlay = _overlay(rgb, candidate, (210, 50, 190))
    agreement = rgb.copy()
    true_positive = oracle & candidate
    false_positive = ~oracle & candidate
    false_negative = oracle & ~candidate
    agreement = _overlay(agreement, true_positive, (30, 210, 80))
    agreement = _overlay(agreement, false_positive, (220, 40, 190))
    agreement = _overlay(agreement, false_negative, (255, 150, 20))
    panels = [
        _panel(ground_truth_overlay, "Ground Truth"),
        _panel(oracle_overlay, "Target-native"),
        _panel(candidate_overlay, candidate_label),
        _panel(agreement, "Agreement G/P/O"),
    ]
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
        "white",
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def write_handoff_artifacts(
    *,
    video_dir: str | Path,
    annotation_dir: str | Path,
    object_id: int,
    oracle_masks: Mapping[int, torch.Tensor],
    candidate_masks: Mapping[int, torch.Tensor],
    output_dir: str | Path,
    report: Mapping[str, Any],
    candidate_label: str,
) -> dict[str, Any]:
    """Write compact PNG previews plus machine- and human-readable reports."""

    video_dir = Path(video_dir).resolve()
    annotation_dir = Path(annotation_dir).resolve()
    output_dir = Path(output_dir).resolve()
    frame_paths = _indexed_frames(video_dir)
    if oracle_masks.keys() != candidate_masks.keys():
        raise ValueError("oracle and candidate frame sets differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir = output_dir / "oracle_masks"
    candidate_dir = output_dir / "candidate_masks"
    comparison_dir = output_dir / "comparisons"
    for directory in (oracle_dir, candidate_dir, comparison_dir):
        directory.mkdir(parents=True, exist_ok=True)

    comparisons: list[str] = []
    for frame in sorted(oracle_masks):
        frame_path = frame_paths.get(frame)
        if frame_path is None:
            raise FileNotFoundError(f"video frame {frame} is unavailable in {video_dir}")
        with Image.open(frame_path) as raw_image:
            image = raw_image.convert("RGB")
        annotation_path = annotation_dir / f"{frame:05d}.png"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"ground-truth annotation is unavailable: {annotation_path}")
        ground_truth = np.asarray(Image.open(annotation_path)) == object_id
        if ground_truth.shape != (image.height, image.width):
            raise ValueError(
                f"ground-truth shape {ground_truth.shape} differs from image "
                f"shape {(image.height, image.width)}"
            )
        oracle = _binary_mask(oracle_masks[frame], image.size)
        candidate = _binary_mask(candidate_masks[frame], image.size)
        name = f"{frame:05d}.png"
        Image.fromarray(oracle.astype(np.uint8) * 255).save(oracle_dir / name)
        Image.fromarray(candidate.astype(np.uint8) * 255).save(candidate_dir / name)
        comparison_name = f"frame_{frame:05d}.png"
        _comparison_canvas(
            image,
            ground_truth,
            oracle,
            candidate,
            candidate_label=candidate_label,
        ).save(comparison_dir / comparison_name)
        comparisons.append(f"comparisons/{comparison_name}")

    artifact_manifest = {
        "directory": ".",
        "oracle_masks": "oracle_masks/",
        "candidate_masks": "candidate_masks/",
        "comparisons": comparisons,
        "report_json": "report.json",
        "report_markdown": "report.md",
    }
    complete_report = dict(report)
    complete_report["artifacts"] = artifact_manifest
    (output_dir / "report.json").write_text(
        json.dumps(complete_report, indent=2), encoding="utf-8"
    )

    comparison = complete_report.get("mask_comparison_to_target_native", {})
    resources = complete_report.get("resources", {})
    markdown = [
        "# SAM 2 handoff preview",
        "",
        f"- Source: `{complete_report.get('source_model_id', 'unknown')}`",
        f"- Target: `{complete_report.get('target_model_id', 'unknown')}`",
        f"- Translator: `{complete_report.get('translator', 'unknown')}`",
        f"- Switch frame: `{complete_report.get('switch_frame', 'unknown')}`",
        f"- Mean binary IoU vs target-native: `{comparison.get('mean_binary_iou', 'n/a')}`",
        f"- Mean mask-logit MSE vs target-native: `{comparison.get('mean_mse', 'n/a')}`",
        f"- Wall time (seconds): `{resources.get('wall_time_seconds', 'n/a')}`",
        f"- Peak CUDA memory (bytes): `{resources.get('peak_cuda_memory_bytes', 'n/a')}`",
        "",
        "The first panel overlays DAVIS ground truth in red. In the agreement panel, "
        "green means both methods selected the pixel, pink is candidate-only, and "
        "orange is Target-native-only.",
        "",
    ]
    for path in comparisons:
        markdown.extend((f"![{path}]({path})", ""))
    (output_dir / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    return artifact_manifest
