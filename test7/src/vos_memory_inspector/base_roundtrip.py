"""Shared same-checkpoint handoff runner and LVOS v2 experiment helpers.

No learned translator is trained or modified here. The existing DirectCopyTranslator
is the identity-valued component of the same-checkpoint handoff. The streaming
predictor, scoring, and curve-writing helpers are shared with the VOST diagnostic.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from .device import resolve_device
from .sam2_state import (
    canonicalize_sam2_inference_state,
    init_sam2_inference_state_without_warmup,
    inject_sam2_canonical_state,
)
from .translators import DirectCopyTranslator
from .upstream import verify_sam2_checkout


# The first four are the original smoke-test cases. The remaining ten are the
# lowest prior Base+ cold-start J&F clips from the local LVOS validation run.
# Their prior score is selection evidence only; it is never used by the gate.
VIDEO_IDS = (
    "2VegYEbT", "2urlAsm8", "0tCWPOrc", "9HEh93ef",
    "x3nD3QQ9", "MKnlVo6x", "8lxxCA5h", "nfcT3owb", "q1MSEBkh",
    "ScFTYisJ", "aFytsETk", "dtHbJvYy", "xpI7xRWN", "K3OUeINk",
)
PRIOR_HARD_JF = {
    "x3nD3QQ9": 0.07730599859907754,
    "MKnlVo6x": 0.27105326832275867,
    "8lxxCA5h": 0.36444947716571596,
    "nfcT3owb": 0.4218655500270669,
    "q1MSEBkh": 0.4242214615036243,
    "ScFTYisJ": 0.542509889298446,
    "aFytsETk": 0.5449554056680113,
    "dtHbJvYy": 0.5515834527119061,
    "xpI7xRWN": 0.5655915945346737,
    "K3OUeINk": 0.5695193303102182,
}
FRAME_COUNT = 40
SWITCH_INDEX = 20
CONFIG_FILE = "configs/sam2.1/sam2.1_hiera_b+.yaml"
EXPECTED_CHECKPOINT_SHA256 = "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
CSV_FIELDS = (
    "video_id", "object_id", "frame_idx", "original_frame_id", "prompt_frame",
    "switch_frame", "phase", "gt_visible", "included_in_mean", "native_J",
    "native_F", "native_JF", "transfer_J", "transfer_F", "transfer_JF",
    "delta_JF", "binary_equal", "logit_max_abs_error",
)
Metric = Callable[..., float]


@dataclass(frozen=True)
class Prompt:
    object_id: int
    frame_idx: int
    original_frame_id: int


@dataclass(frozen=True)
class Clip:
    video_id: str
    frames: tuple[Path, ...]
    annotations: tuple[Path, ...]
    original_frame_ids: tuple[int, ...]
    prompts: tuple[Prompt, ...]
    switch_index: int = SWITCH_INDEX

    @property
    def prompt_indices(self) -> set[int]:
        return {prompt.frame_idx for prompt in self.prompts}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_downloads(data_root: Path) -> dict[str, Any]:
    """Check every selected local file against its official-archive entry digest."""

    manifest = json.loads((data_root / "download_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != "LVOS v2 validation" or tuple(manifest.get("videos", ())) != VIDEO_IDS:
        raise ValueError("unexpected LVOS download manifest")
    files = manifest.get("files", [])
    expected_files = len(VIDEO_IDS) * FRAME_COUNT * 2
    if len(files) != expected_files:
        raise ValueError(f"expected {expected_files} selected files, found {len(files)}")
    seen: set[str] = set()
    total_bytes = 0
    for entry in files:
        name = str(entry["filename"])
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in seen:
            raise ValueError(f"invalid or duplicate download path: {name}")
        seen.add(name)
        path = data_root / relative
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"wrong byte count: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        total_bytes += path.stat().st_size
    return {
        "files": len(files), "bytes": total_bytes, "videos": list(VIDEO_IDS),
        "hard_video_prior_JF": PRIOR_HARD_JF,
    }


def load_clip(
    data_root: Path,
    manifest: dict[str, Any],
    video_id: str,
    *,
    frame_count: int = FRAME_COUNT,
    switch_index: int = SWITCH_INDEX,
) -> Clip:
    """Validate paired sparse LVOS frames and derive first-mask prompts only."""

    if manifest.get("schema_version") != "cmmt.lvosv2_evaluation_manifest.v1":
        raise ValueError("LVOS v2 evaluation manifest is required")
    if manifest.get("split") != "val":
        raise ValueError("only LVOS v2 validation is supported")
    if frame_count <= switch_index + 1 or switch_index < 0:
        raise ValueError("switch must have a prefix and a future continuation")
    image_dir = data_root / "JPEGImages" / video_id
    annotation_dir = data_root / "Annotations" / video_id
    frames = sorted(image_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    if len(frames) != frame_count:
        raise ValueError(f"{video_id}: expected exactly {frame_count} RGB frames, got {len(frames)}")
    frame_ids = tuple(int(path.stem) for path in frames)
    if len(set(frame_ids)) != frame_count:
        raise ValueError(f"{video_id}: duplicate original frame IDs")
    annotations = tuple(annotation_dir / f"{path.stem}.png" for path in frames)
    if any(not path.is_file() for path in annotations):
        raise FileNotFoundError(f"{video_id}: RGB/annotation frame IDs differ")
    extra_annotations = {path.stem for path in annotation_dir.glob("*.png")} - {
        path.stem for path in frames
    }
    if extra_annotations:
        raise ValueError(f"{video_id}: unexpected annotation IDs: {sorted(extra_annotations)}")

    first_by_object: dict[int, int] = {}
    for case in manifest.get("cases", []):
        if case.get("video_id") != video_id:
            continue
        object_id = int(case["object_id"])
        first_frame = int(case["first_prompt_frame"])
        if object_id in first_by_object and first_by_object[object_id] != first_frame:
            raise ValueError(f"{video_id}: conflicting first prompt for object {object_id}")
        first_by_object[object_id] = first_frame
    if not first_by_object:
        raise ValueError(f"{video_id}: no objects in the LVOS manifest")
    prompts: list[Prompt] = []
    for object_id, original_frame_id in first_by_object.items():
        if original_frame_id not in frame_ids:
            raise ValueError(f"{video_id}: object {object_id} first prompt is outside the clip")
        frame_idx = frame_ids.index(original_frame_id)
        if frame_idx > switch_index:
            raise ValueError(f"{video_id}: object {object_id} is prompted after switch")
        with Image.open(annotations[frame_idx]) as image:
            labels = np.asarray(image)
        if not np.any(labels == object_id):
            raise ValueError(f"{video_id}: object {object_id} absent in its first prompt mask")
        prompts.append(Prompt(object_id, frame_idx, original_frame_id))
    return Clip(
        video_id=video_id,
        frames=tuple(frames),
        annotations=annotations,
        original_frame_ids=frame_ids,
        prompts=tuple(sorted(prompts, key=lambda p: (p.frame_idx, p.object_id))),
        switch_index=switch_index,
    )


def load_official_lvos_metrics(repository: Path) -> tuple[Metric, Metric, str]:
    """Import J/F implementations from a checked-out official LVOS evaluator."""

    repository = repository.resolve()
    if not (repository / "lvos" / "metrics.py").is_file():
        raise FileNotFoundError(f"official lvos-evaluation checkout missing: {repository}")
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    try:
        from lvos.metrics import db_eval_boundary, db_eval_iou
    except ImportError as exc:
        raise RuntimeError(
            "official LVOS metrics require opencv-python-headless and scikit-image"
        ) from exc
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return db_eval_iou, db_eval_boundary, commit


def masks_by_id(
    object_ids: list[int] | tuple[int, ...], masks: torch.Tensor
) -> tuple[dict[int, torch.Tensor], dict[int, np.ndarray]]:
    """Produce mutually exclusive per-object masks from SAM 2 video logits."""

    ids = tuple(int(value) for value in object_ids)
    if len(ids) == 0 or len(ids) != len(set(ids)):
        raise ValueError(f"invalid predictor object IDs: {ids}")
    values = masks.detach().cpu().float()
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3 or values.shape[0] != len(ids):
        raise ValueError(f"unexpected video mask shape {tuple(values.shape)} for {ids}")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("non-finite SAM 2 mask logits")
    logits = {object_id: values[index] for index, object_id in enumerate(ids)}
    foreground_scores, owner = values.max(dim=0)
    labelmap = torch.zeros_like(owner, dtype=torch.int32)
    for index, object_id in enumerate(ids):
        labelmap[(owner == index) & (foreground_scores > 0)] = object_id
    binary = {object_id: (labelmap == object_id).numpy() for object_id in ids}
    return logits, binary


def score_frame(
    clip: Clip,
    frame_idx: int,
    object_ids: list[int] | tuple[int, ...],
    masks: torch.Tensor,
    iou_metric: Metric,
    boundary_metric: Metric,
) -> tuple[dict[int, dict[str, float | bool]], dict[int, torch.Tensor], dict[int, np.ndarray]]:
    logits, binary = masks_by_id(object_ids, masks)
    with Image.open(clip.annotations[frame_idx]) as image:
        labels = np.asarray(image)
    if labels.ndim != 2:
        raise ValueError(f"{clip.video_id}: GT is not a single-channel label mask")
    if labels.shape != next(iter(binary.values())).shape:
        raise ValueError(f"{clip.video_id}: prediction and GT image size differ")
    void = labels == 255
    scores: dict[int, dict[str, float | bool]] = {}
    for object_id, prediction in binary.items():
        annotation = labels == object_id
        j = float(iou_metric(annotation, prediction, void_pixels=void))
        f = float(boundary_metric(annotation, prediction, void_pixels=void))
        if not np.isfinite(j) or not np.isfinite(f):
            raise ValueError(f"non-finite J/F for {clip.video_id} frame {frame_idx}")
        scores[object_id] = {"J": j, "F": f, "JF": (j + f) / 2, "visible": bool(annotation.any())}
    return scores, logits, binary


def make_frame_rows(
    clip: Clip,
    frame_idx: int,
    native: tuple[dict[int, dict[str, float | bool]], dict[int, torch.Tensor], dict[int, np.ndarray]],
    transferred: tuple[dict[int, dict[str, float | bool]], dict[int, torch.Tensor], dict[int, np.ndarray]] | None,
) -> list[dict[str, Any]]:
    native_scores, native_logits, native_binary = native
    if transferred is None:
        transferred = native  # The two methods deliberately share the same prefix.
    target_scores, target_logits, target_binary = transferred
    if native_scores.keys() != target_scores.keys():
        raise ValueError(f"{clip.video_id} frame {frame_idx}: object IDs changed at handoff")
    phase = "post_switch" if frame_idx > clip.switch_index else (
        "switch" if frame_idx == clip.switch_index else "pre_switch"
    )
    rows: list[dict[str, Any]] = []
    for object_id, source in native_scores.items():
        target = target_scores[object_id]
        prompt_frame = next(p.frame_idx for p in clip.prompts if p.object_id == object_id)
        max_error = float((native_logits[object_id] - target_logits[object_id]).abs().max())
        rows.append({
            "video_id": clip.video_id,
            "object_id": object_id,
            "frame_idx": frame_idx,
            "original_frame_id": clip.original_frame_ids[frame_idx],
            "prompt_frame": frame_idx == prompt_frame,
            "switch_frame": frame_idx == clip.switch_index,
            "phase": phase,
            "gt_visible": source["visible"],
            "included_in_mean": frame_idx != prompt_frame,
            "native_J": source["J"], "native_F": source["F"], "native_JF": source["JF"],
            "transfer_J": target["J"], "transfer_F": target["F"], "transfer_JF": target["JF"],
            "delta_JF": float(target["JF"]) - float(source["JF"]),
            "binary_equal": bool(np.array_equal(native_binary[object_id], target_binary[object_id])),
            "logit_max_abs_error": max_error,
        })
    return rows


def _assert_identity_transfer(source: Any, translated: Any) -> None:
    for name in (
        "spatial_memory", "object_pointer", "presence_logits", "frame_indices", "slot_order",
        "is_conditioning", "validity",
    ):
        if not torch.equal(getattr(source, name), getattr(translated, name)):
            raise ValueError(f"identity translator changed {name}")
    if source.object_ids != translated.object_ids or source.switch_frame != translated.switch_frame:
        raise ValueError("identity translator changed object IDs or switch frame")


def _write_prediction_labelmap(
    path: Path, binary_masks: dict[int, np.ndarray]
) -> None:
    """Write object-ID labels in indexed-PNG form for the official VOST evaluator."""

    if not binary_masks:
        raise ValueError("cannot write a prediction without object masks")
    first = next(iter(binary_masks.values()))
    labelmap = np.zeros(first.shape, dtype=np.uint8)
    for object_id, mask in binary_masks.items():
        if not 0 < int(object_id) < 255:
            raise ValueError(f"object ID cannot be represented in VOST PNG: {object_id}")
        if mask.shape != labelmap.shape:
            raise ValueError("VOST prediction masks have inconsistent dimensions")
        labelmap[mask] = int(object_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labelmap).save(path)


@torch.inference_mode()
def run_video(
    clip: Clip,
    *,
    sam2_repo: Path,
    checkpoint: Path,
    device: str,
    iou_metric: Metric,
    boundary_metric: Metric,
    seed: int = 7,
    logit_tolerance: float = 1e-6,
    metric_tolerance: float = 1e-6,
    prediction_roots: dict[str, Path] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream native and injected continuations in lockstep; never cache video logits."""

    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    if prediction_roots is not None and set(prediction_roots) != {"native", "transferred"}:
        raise ValueError("prediction_roots must contain exactly native and transferred outputs")
    from sam2.build_sam import build_sam2_video_predictor

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    source_predictor = build_sam2_video_predictor(CONFIG_FILE, str(checkpoint), device=device)
    source_state = source_predictor.init_state(
        video_path=str(clip.frames[0].parent),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    if int(source_state["num_frames"]) != len(clip.frames):
        raise ValueError(f"{clip.video_id}: SAM 2 saw the wrong number of frames")

    by_frame: dict[int, list[Prompt]] = {}
    for prompt in clip.prompts:
        by_frame.setdefault(prompt.frame_idx, []).append(prompt)
    rows: list[dict[str, Any]] = []
    target_predictor = None
    target_state = None
    target_iterator = None
    injection: dict[str, Any] | None = None
    backbone_calls_during_injection: int | None = None
    target_backbone_calls: list[int] = []
    handoff_bytes: int | None = None
    prompt_frames = sorted(by_frame)

    for segment, prompt_frame in enumerate(prompt_frames):
        for prompt in by_frame[prompt_frame]:
            with Image.open(clip.annotations[prompt.frame_idx]) as image:
                labels = np.asarray(image)
            source_predictor.add_new_mask(
                source_state,
                frame_idx=prompt.frame_idx,
                obj_id=prompt.object_id,
                mask=labels == prompt.object_id,
            )
        end = prompt_frames[segment + 1] if segment + 1 < len(prompt_frames) else len(clip.frames)
        for frame_idx, object_ids, masks in source_predictor.propagate_in_video(
            source_state,
            start_frame_idx=prompt_frame,
            max_frame_num_to_track=end - prompt_frame - 1,
            reverse=False,
        ):
            frame_idx = int(frame_idx)
            if not prompt_frame <= frame_idx < end:
                raise ValueError(f"{clip.video_id}: unexpected native frame {frame_idx}")
            native = score_frame(clip, frame_idx, object_ids, masks, iou_metric, boundary_metric)
            if prediction_roots is not None:
                filename = f"{clip.annotations[frame_idx].stem}.png"
                _write_prediction_labelmap(
                    prediction_roots["native"] / clip.video_id / filename, native[2]
                )
            if frame_idx == clip.switch_index:
                canonical = canonicalize_sam2_inference_state(
                    source_state, switch_frame=frame_idx, strict=True
                )
                translated = DirectCopyTranslator(canonical.spec).translate(canonical)
                _assert_identity_transfer(canonical, translated)
                handoff_bytes = translated.handoff_bytes()
                target_predictor = build_sam2_video_predictor(
                    CONFIG_FILE, str(checkpoint), device=device
                )
                original_forward_image = target_predictor.forward_image

                def counted_forward_image(image: torch.Tensor) -> Any:
                    target_backbone_calls.append(1)
                    return original_forward_image(image)

                target_predictor.forward_image = counted_forward_image
                target_state = init_sam2_inference_state_without_warmup(
                    target_predictor,
                    video_path=str(clip.frames[0].parent),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                )
                before_injection = len(target_backbone_calls)
                injection = inject_sam2_canonical_state(
                    translated, predictor=target_predictor, inference_state=target_state
                )
                backbone_calls_during_injection = len(target_backbone_calls) - before_injection
                if backbone_calls_during_injection != 0:
                    raise RuntimeError("target recomputed past RGB during injection")
                if tuple(target_state["obj_ids"]) != canonical.object_ids:
                    raise ValueError("Target did not reconstruct the registered object IDs")
                target_iterator = iter(target_predictor.propagate_in_video(
                    target_state,
                    start_frame_idx=clip.switch_index + 1,
                    max_frame_num_to_track=len(clip.frames) - clip.switch_index - 1,
                    reverse=False,
                ))
            if frame_idx > clip.switch_index:
                if target_iterator is None:
                    raise RuntimeError("target continuation was not initialized")
                target_frame, target_ids, target_masks = next(target_iterator)
                if int(target_frame) != frame_idx:
                    raise ValueError(f"native/target frame mismatch: {frame_idx}, {target_frame}")
                transferred = score_frame(
                    clip, frame_idx, target_ids, target_masks, iou_metric, boundary_metric
                )
                if prediction_roots is not None:
                    filename = f"{clip.annotations[frame_idx].stem}.png"
                    _write_prediction_labelmap(
                        prediction_roots["transferred"] / clip.video_id / filename,
                        transferred[2],
                    )
                rows.extend(make_frame_rows(clip, frame_idx, native, transferred))
            else:
                if prediction_roots is not None:
                    filename = f"{clip.annotations[frame_idx].stem}.png"
                    _write_prediction_labelmap(
                        prediction_roots["transferred"] / clip.video_id / filename, native[2]
                    )
                rows.extend(make_frame_rows(clip, frame_idx, native, None))

    if target_iterator is None or injection is None:
        raise RuntimeError(f"{clip.video_id}: native run never reached the switch frame")
    if next(target_iterator, None) is not None:
        raise ValueError(f"{clip.video_id}: Target emitted extra frames")
    post = [row for row in rows if row["phase"] == "post_switch" and row["included_in_mean"]]
    if not post:
        raise ValueError(f"{clip.video_id}: no post-switch objects were scored")
    expected_rows = sum(len([p for p in clip.prompts if p.frame_idx <= frame]) for frame in range(len(clip.frames)))
    if len(rows) != expected_rows:
        raise ValueError(f"{clip.video_id}: incomplete object/frame rows")
    first = [row for row in post if row["frame_idx"] == clip.switch_index + 1]
    max_error = max(row["logit_max_abs_error"] for row in post)
    max_metric_delta = max(
        abs(row[name]) for row in post
        for name in ("delta_JF",)
    )
    expected_target_backbone_calls = len(clip.frames) - clip.switch_index - 1
    target_backbone_calls_total = len(target_backbone_calls)
    passed = bool(
        backbone_calls_during_injection == 0
        and target_backbone_calls_total == expected_target_backbone_calls
        and all(row["binary_equal"] for row in post)
        and max_error <= logit_tolerance
        and all(
            abs(row[f"native_{name}"] - row[f"transfer_{name}"]) <= metric_tolerance
            for row in post for name in ("J", "F", "JF")
        )
    )
    first_mismatch = next((
        {"frame_idx": row["frame_idx"], "original_frame_id": row["original_frame_id"],
         "object_id": row["object_id"], "binary_equal": row["binary_equal"],
         "logit_max_abs_error": row["logit_max_abs_error"], "delta_JF": row["delta_JF"]}
        for row in post if not row["binary_equal"] or row["logit_max_abs_error"] > logit_tolerance
        or any(abs(row[f"native_{name}"] - row[f"transfer_{name}"]) > metric_tolerance
               for name in ("J", "F", "JF"))
    ), None)
    summary = {
        "video_id": clip.video_id,
        "object_ids": [prompt.object_id for prompt in clip.prompts],
        "prompt_frames": [prompt.__dict__ for prompt in clip.prompts],
        "frame_ids": list(clip.original_frame_ids),
        "switch_index": clip.switch_index,
        "switch_original_frame_id": clip.original_frame_ids[clip.switch_index],
        "scored_rows": len(rows),
        "post_switch_rows": len(post),
        "first_post_switch_mean_delta_JF": mean(row["delta_JF"] for row in first),
        "post_switch_mean_delta_JF": mean(row["delta_JF"] for row in post),
        "post_switch_min_delta_JF": min(row["delta_JF"] for row in post),
        "post_switch_max_delta_JF": max(row["delta_JF"] for row in post),
        "post_switch_max_abs_delta_JF": max_metric_delta,
        "post_switch_max_logit_error": max_error,
        "binary_equal_all": all(row["binary_equal"] for row in post),
        "injection": injection,
        "handoff_bytes": handoff_bytes,
        "backbone_calls_during_injection": backbone_calls_during_injection,
        "backbone_calls_during_target_continuation": target_backbone_calls_total,
        "expected_backbone_calls_during_target_continuation": expected_target_backbone_calls,
        "first_mismatch": first_mismatch,
        "passed": passed,
    }
    del target_state, target_predictor, source_state, source_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, summary


def _polyline(
    points: list[tuple[int, float]], top: int, frame_count: int, *, color: str, dashed: bool
) -> str:
    if not points:
        return ""
    last_frame = max(frame_count - 1, 1)
    coords = " ".join(
        f"{90 + x * 600 / last_frame:.2f},{top + 125 - y * 100:.2f}"
        for x, y in points
    )
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dash}/>'


def write_svg(clip: Clip, rows: list[dict[str, Any]], path: Path) -> None:
    """Write a dependency-free SVG with one panel per object and an active-object mean."""

    object_ids = [prompt.object_id for prompt in clip.prompts]
    panels: list[tuple[str, list[dict[str, Any]], int | None]] = [
        (f"Object {object_id}", [row for row in rows if row["object_id"] == object_id],
         next(p.frame_idx for p in clip.prompts if p.object_id == object_id))
        for object_id in object_ids
    ]
    mean_rows: list[dict[str, Any]] = []
    for frame_idx in range(len(clip.frames)):
        frame_rows = [row for row in rows if row["frame_idx"] == frame_idx]
        if frame_rows:
            mean_rows.append({
                "frame_idx": frame_idx,
                "native_JF": mean(row["native_JF"] for row in frame_rows),
                "transfer_JF": mean(row["transfer_JF"] for row in frame_rows),
            })
    panels.append(("Mean of prompted objects", mean_rows, None))
    height = 45 + len(panels) * 175
    last_frame = max(len(clip.frames) - 1, 1)
    frame_x = lambda frame_idx: 90 + frame_idx * 600 / last_frame
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{height}" viewBox="0 0 760 {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="24" font-size="18">{escape(clip.video_id)} — J&amp;F by clip frame</text>',
    ]
    for index, (title, panel_rows, prompt_idx) in enumerate(panels):
        top = 35 + index * 175
        content.extend([
            f'<text x="20" y="{top + 18}" font-size="14">{escape(title)}</text>',
            f'<path d="M90 {top + 25} V{top + 125} H690" fill="none" stroke="#888"/>',
            f'<text x="65" y="{top + 30}" font-size="11">1.0</text>',
            f'<text x="65" y="{top + 128}" font-size="11">0.0</text>',
            f'<line x1="{frame_x(clip.switch_index):.2f}" x2="{frame_x(clip.switch_index):.2f}" '
            f'y1="{top + 25}" y2="{top + 125}" stroke="#d23b3b" stroke-width="1.5"/>',
            f'<text x="90" y="{top + 145}" font-size="11">0</text>',
            f'<text x="{frame_x(clip.switch_index) - 22:.2f}" y="{top + 145}" '
            f'font-size="11">switch {clip.switch_index}</text>',
            f'<text x="{frame_x(last_frame) - 8:.2f}" y="{top + 145}" '
            f'font-size="11">{last_frame}</text>',
        ])
        prompt_markers = [prompt_idx] if prompt_idx is not None else sorted(clip.prompt_indices)
        for marker in prompt_markers:
            content.append(
                f'<line x1="{frame_x(marker):.2f}" x2="{frame_x(marker):.2f}" '
                f'y1="{top + 25}" y2="{top + 125}" stroke="#239b6b" stroke-dasharray="2 4"/>'
            )
        content.append(_polyline(
            [(row["frame_idx"], row["native_JF"]) for row in panel_rows],
            top, len(clip.frames), color="#1768ac", dashed=False,
        ))
        content.append(_polyline(
            [(row["frame_idx"], row["transfer_JF"]) for row in panel_rows],
            top, len(clip.frames), color="#e57a19", dashed=True,
        ))
    content.extend([
        f'<text x="18" y="{height - 9}" font-size="12" fill="#1768ac">Native — solid</text>',
        f'<text x="150" y="{height - 9}" font-size="12" fill="#e57a19">Transferred — dashed</text>',
        f'<text x="330" y="{height - 9}" font-size="12" fill="#239b6b">Prompt</text>',
        f'<text x="430" y="{height - 9}" font-size="12" fill="#d23b3b">Switch</text>',
        '</svg>',
    ])
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def write_video_artifacts(
    clip: Clip, rows: list[dict[str, Any]], summary: dict[str, Any], output_root: Path
) -> None:
    target = output_root / clip.video_id
    target.mkdir(parents=True, exist_ok=False)
    with (target / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (target / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_svg(clip, rows, target / "jf_curve.svg")


def run_experiment(
    *,
    data_root: Path,
    manifest_path: Path,
    sam2_repo: Path,
    checkpoint: Path,
    metrics_repo: Path,
    output_root: Path,
    device: str = "auto",
    seed: int = 7,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {output_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = [load_clip(data_root, manifest, video_id) for video_id in VIDEO_IDS]
    download_validation = validate_downloads(data_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"unexpected Base+ checkpoint SHA-256 {checkpoint_sha256}; "
            f"expected {EXPECTED_CHECKPOINT_SHA256}"
        )
    upstream_commit = verify_sam2_checkout(sam2_repo)
    iou_metric, boundary_metric, evaluator_commit = load_official_lvos_metrics(metrics_repo)
    actual_device = resolve_device(device, allow_mps=False)
    output_root.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": "cmmt.lvos_base_roundtrip.v1",
        "status": "running",
        "claim": "LVOS v2 fourteen-clip same-checkpoint implementation diagnostic; not official dataset score",
        "manifest_usage": (
            "first_prompt_frame only; the switch is fixed at clip index 20 for every video "
            "and is not one of the frozen manifest switch cases"
        ),
        "scoring_policy": (
            "official per-frame J/F; prompt frame excluded; scored through the clip end "
            "without the official frame_range end or last-frame exclusion"
        ),
        "cuda_offload_path_exercised": actual_device == "cuda",
        "translator": "DirectCopyTranslator",
        "videos": list(VIDEO_IDS),
        "hard_video_prior_JF": PRIOR_HARD_JF,
        "frame_count": FRAME_COUNT,
        "switch_index": SWITCH_INDEX,
        "seed": seed,
        "device": actual_device,
        "sam2_commit": upstream_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": sha256_file(manifest_path),
        "download_manifest_sha256": sha256_file(data_root / "download_manifest.json"),
        "download_validation": download_validation,
        "lvos_evaluator_commit": evaluator_commit,
        "results": [],
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for clip in clips:
        try:
            rows, summary = run_video(
                clip,
                sam2_repo=sam2_repo,
                checkpoint=checkpoint,
                device=actual_device,
                iou_metric=iou_metric,
                boundary_metric=boundary_metric,
                seed=seed,
            )
            write_video_artifacts(clip, rows, summary, output_root)
            report["results"].append(summary)
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = {"video_id": clip.video_id, "type": type(exc).__name__, "message": str(exc)}
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            raise
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["status"] = "passed" if all(row["passed"] for row in report["results"]) else "failed_gate"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
