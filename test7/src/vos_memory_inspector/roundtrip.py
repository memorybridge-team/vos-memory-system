"""Checkpoint-backed same-model export/inject continuation validation."""

from __future__ import annotations

import gc
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .device import resolve_device
from .runner import load_binary_prompt
from .artifacts import write_handoff_artifacts
from .case_cache import load_case_cache, write_case_cache
from .metrics import evaluate_state
from .sam2_state import (
    canonicalize_sam2_inference_state,
    init_sam2_inference_state_without_warmup,
    inject_sam2_canonical_state,
)
from .upstream import verify_sam2_checkout
from .translators import DirectCopyTranslator


CACHED_BASELINES = ("target_reset", "last_mask", "replay_k", "full_replay")


@dataclass(frozen=True)
class MaskPromptEvent:
    """One user-provided object mask at a specific video frame."""

    frame_index: int
    object_id: int
    mask_path: Path


def _validate_prompt_events(
    prompt_events: list[MaskPromptEvent], *, switch_frame: int, num_frames: int
) -> list[MaskPromptEvent]:
    if not prompt_events:
        raise ValueError("at least one mask prompt event is required")
    ordered = sorted(prompt_events, key=lambda event: (event.frame_index, event.object_id))
    seen: set[tuple[int, int]] = set()
    for event in ordered:
        identity = (event.frame_index, event.object_id)
        if identity in seen:
            raise ValueError(f"duplicate prompt event frame/object: {identity}")
        seen.add(identity)
        if not 0 <= event.frame_index <= switch_frame:
            raise ValueError(
                f"prompt frame {event.frame_index} must be between 0 and switch "
                f"frame {switch_frame}"
            )
        if event.frame_index >= num_frames:
            raise ValueError(
                f"prompt frame {event.frame_index} exceeds video length {num_frames}"
            )
        if not event.mask_path.is_file():
            raise FileNotFoundError(f"prompt mask not found: {event.mask_path}")
    return ordered


def plan_correction_route(*, correction_frame: int, switch_frame: int) -> dict[str, Any]:
    """Choose the supported correction path for the minimal handoff contract.

    A correction after the switch belongs to the Target runtime and can be applied
    directly.  A correction at or before the switch cannot refine the injected
    history because that history intentionally excludes past ``pred_masks``; it
    must rebuild Target-native history from the recorded interaction timeline.
    """

    if switch_frame < 0:
        raise ValueError("switch_frame must be non-negative")
    if correction_frame < 0:
        raise ValueError("correction_frame must be non-negative")
    if correction_frame > switch_frame:
        return {
            "mode": "target_current_frame",
            "requires_history_replay": False,
            "correction_frame": correction_frame,
            "switch_frame": switch_frame,
        }
    return {
        "mode": "target_replay_from_prompt_anchor",
        "requires_history_replay": True,
        "correction_frame": correction_frame,
        "switch_frame": switch_frame,
    }


def _validate_timeline_events(
    prompt_events: list[MaskPromptEvent], *, max_frame: int, num_frames: int
) -> list[MaskPromptEvent]:
    """Validate a complete Target interaction timeline through ``max_frame``."""

    if not prompt_events:
        raise ValueError("at least one mask prompt event is required")
    ordered = sorted(prompt_events, key=lambda event: (event.frame_index, event.object_id))
    seen: set[tuple[int, int]] = set()
    for event in ordered:
        identity = (event.frame_index, event.object_id)
        if identity in seen:
            raise ValueError(f"duplicate prompt event frame/object: {identity}")
        seen.add(identity)
        if not 0 <= event.frame_index <= max_frame:
            raise ValueError(
                f"prompt frame {event.frame_index} must be between 0 and "
                f"frame {max_frame}"
            )
        if event.frame_index >= num_frames:
            raise ValueError(
                f"prompt frame {event.frame_index} exceeds video length {num_frames}"
            )
        if not event.mask_path.is_file():
            raise FileNotFoundError(f"prompt mask not found: {event.mask_path}")
    return ordered


def _run_mask_prompt_timeline(
    predictor: Any,
    inference_state: dict[str, Any],
    *,
    prompt_events: list[MaskPromptEvent],
    final_frame: int,
) -> dict[int, torch.Tensor]:
    """Run an explicit mask-prompt timeline and return logits through a frame."""

    events = _validate_timeline_events(
        prompt_events,
        max_frame=final_frame,
        num_frames=int(inference_state["num_frames"]),
    )
    events_by_frame: dict[int, list[MaskPromptEvent]] = {}
    for event in events:
        events_by_frame.setdefault(event.frame_index, []).append(event)

    outputs: dict[int, torch.Tensor] = {}
    cursor = min(events_by_frame)
    for prompt_frame in sorted(events_by_frame):
        if cursor < prompt_frame:
            for frame_idx, _object_ids, masks in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=cursor,
                max_frame_num_to_track=prompt_frame - cursor,
                reverse=False,
            ):
                frame = int(frame_idx)
                if frame < prompt_frame:
                    outputs[frame] = masks.detach().cpu().float()
        latest_masks = None
        for event in events_by_frame[prompt_frame]:
            _frame_idx, _object_ids, latest_masks = predictor.add_new_mask(
                inference_state,
                frame_idx=event.frame_index,
                obj_id=event.object_id,
                mask=load_binary_prompt(event.mask_path, event.object_id),
            )
        if latest_masks is not None:
            outputs[prompt_frame] = latest_masks.detach().cpu().float()
        cursor = prompt_frame

    for frame_idx, _object_ids, masks in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=cursor,
        max_frame_num_to_track=final_frame - cursor + 1,
        reverse=False,
    ):
        frame = int(frame_idx)
        if frame <= final_frame:
            outputs[frame] = masks.detach().cpu().float()
    return outputs


def _collect_native_prompt_timeline(
    predictor: Any,
    inference_state: dict[str, Any],
    *,
    prompt_events: list[MaskPromptEvent],
    switch_frame: int,
) -> tuple[Any, dict[int, torch.Tensor]]:
    """Run a prompt timeline and collect native state plus post-switch masks."""

    events = _validate_prompt_events(
        prompt_events,
        switch_frame=switch_frame,
        num_frames=int(inference_state["num_frames"]),
    )
    events_by_frame: dict[int, list[MaskPromptEvent]] = {}
    for event in events:
        events_by_frame.setdefault(event.frame_index, []).append(event)

    canonical = None
    future: dict[int, torch.Tensor] = {}
    cursor = min(events_by_frame)
    for prompt_frame in sorted(events_by_frame):
        if cursor < prompt_frame:
            for _frame_idx, _object_ids, _masks in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=cursor,
                max_frame_num_to_track=prompt_frame - cursor - 1,
                reverse=False,
            ):
                pass
        for event in events_by_frame[prompt_frame]:
            predictor.add_new_mask(
                inference_state,
                frame_idx=event.frame_index,
                obj_id=event.object_id,
                mask=load_binary_prompt(event.mask_path, event.object_id),
            )
        cursor = prompt_frame

    for frame_idx, _object_ids, masks in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=cursor,
        max_frame_num_to_track=int(inference_state["num_frames"]) - cursor,
        reverse=False,
    ):
        frame = int(frame_idx)
        if frame == switch_frame:
            canonical = canonicalize_sam2_inference_state(
                inference_state,
                switch_frame=switch_frame,
                strict=True,
            )
        if frame > switch_frame:
            future[frame] = masks.detach().cpu().float()
    if canonical is None:
        raise RuntimeError("native prompt-timeline run did not reach switch_frame")
    return canonical, future


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _start_resource_measurement(device: str) -> float:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def _resource_measurement(started_at: float, device: str) -> dict[str, Any]:
    peak_cuda_memory = None
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        peak_cuda_memory = int(torch.cuda.max_memory_allocated())
    return {
        "wall_time_seconds": float(time.perf_counter() - started_at),
        "peak_cuda_memory_bytes": peak_cuda_memory,
    }


def _collect_native(
    predictor: Any,
    inference_state: dict[str, Any],
    *,
    mask: np.ndarray,
    object_id: int,
    switch_frame: int,
) -> tuple[Any, dict[int, torch.Tensor]]:
    predictor.add_new_mask(
        inference_state,
        frame_idx=0,
        obj_id=object_id,
        mask=mask,
    )
    canonical = None
    future: dict[int, torch.Tensor] = {}
    for frame_idx, _object_ids, masks in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=int(inference_state["num_frames"]),
        reverse=False,
    ):
        frame = int(frame_idx)
        if frame == switch_frame:
            canonical = canonicalize_sam2_inference_state(
                inference_state,
                switch_frame=switch_frame,
                strict=True,
            )
        if frame > switch_frame:
            future[frame] = masks.detach().cpu().float()
    if canonical is None:
        raise RuntimeError("native run did not reach switch_frame")
    return canonical, future


def _collect_prefix(
    predictor: Any,
    inference_state: dict[str, Any],
    *,
    mask: np.ndarray,
    object_id: int,
    switch_frame: int,
):
    canonical, _masks = _collect_prefix_reference(
        predictor,
        inference_state,
        mask=mask,
        object_id=object_id,
        switch_frame=switch_frame,
    )
    return canonical


def _collect_prefix_reference(
    predictor: Any,
    inference_state: dict[str, Any],
    *,
    mask: np.ndarray,
    object_id: int,
    switch_frame: int,
) -> tuple[Any, dict[int, torch.Tensor]]:
    predictor.add_new_mask(
        inference_state,
        frame_idx=0,
        obj_id=object_id,
        mask=mask,
    )
    prefix_masks: dict[int, torch.Tensor] = {}
    canonical = None
    for frame_idx, _object_ids, masks in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=switch_frame,
        reverse=False,
    ):
        frame = int(frame_idx)
        prefix_masks[frame] = masks.detach().cpu().float()
        if frame == switch_frame:
            canonical = canonicalize_sam2_inference_state(
                inference_state,
                switch_frame=switch_frame,
                strict=True,
            )
    if canonical is None:
        raise RuntimeError("source run did not reach switch_frame")
    return canonical, prefix_masks


def plan_cached_baseline(
    baseline: str,
    *,
    switch_frame: int,
    replay_frames: int | None = None,
) -> dict[str, Any]:
    """Resolve a baseline into its prompt source and target history window.

    ``replay_frames`` counts frames through and including the switch frame.  Under
    this definition Replay-1 intentionally equals Last-Mask and serves as an
    implementation sanity check.
    """

    if baseline not in CACHED_BASELINES:
        raise ValueError(f"unsupported cached baseline: {baseline!r}")
    if switch_frame < 0:
        raise ValueError("switch_frame must be non-negative")
    if baseline == "target_reset":
        return {
            "baseline": baseline,
            "prompt_frame": switch_frame,
            "prompt_source": "blank_mask",
            "history_frames_reprocessed": 1,
            "uses_ground_truth_prompt": False,
            "uses_source_prediction_prompt": False,
        }
    if baseline == "last_mask":
        return {
            "baseline": baseline,
            "prompt_frame": switch_frame,
            "prompt_source": "source_prediction_at_switch",
            "history_frames_reprocessed": 1,
            "uses_ground_truth_prompt": False,
            "uses_source_prediction_prompt": True,
        }
    if baseline == "replay_k":
        if replay_frames is None:
            raise ValueError("replay_frames is required for replay_k")
        if replay_frames < 1:
            raise ValueError("replay_frames must be at least 1")
        if replay_frames > switch_frame + 1:
            raise ValueError(
                f"replay_frames={replay_frames} exceeds available prefix "
                f"length {switch_frame + 1}"
            )
        return {
            "baseline": baseline,
            "prompt_frame": switch_frame - replay_frames + 1,
            "prompt_source": "source_prediction_at_replay_start",
            "history_frames_reprocessed": replay_frames,
            "uses_ground_truth_prompt": False,
            "uses_source_prediction_prompt": True,
        }
    return {
        "baseline": baseline,
        "prompt_frame": 0,
        "prompt_source": "first_frame_ground_truth",
        "history_frames_reprocessed": switch_frame + 1,
        "uses_ground_truth_prompt": True,
        "uses_source_prediction_prompt": False,
    }


def binary_prompt_from_mask_logits(
    logits: torch.Tensor,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Convert cached one-object logits to an original-resolution binary prompt."""

    array = logits.detach().cpu().float().numpy().squeeze()
    if array.ndim != 2:
        raise ValueError(f"expected one-object 2D mask logits, got {array.shape}")
    image = Image.fromarray((array > 0).astype(np.uint8) * 255)
    if image.size != (width, height):
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def prepare_cross_model_case_reference(
    *,
    sam2_repo: str | Path,
    source_config_file: str,
    source_checkpoint: str | Path,
    source_model_id: str,
    target_config_file: str,
    target_checkpoint: str | Path,
    target_model_id: str,
    video_dir: str | Path,
    prompt_mask: str | Path,
    object_id: int,
    switch_frame: int,
    output: str | Path,
    device: str | None = None,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
) -> dict[str, Any]:
    """Compute source prefix and target oracle once for reuse by all baselines."""

    device = resolve_device(device, allow_mps=False)
    sam2_repo = Path(sam2_repo).resolve()
    source_checkpoint = Path(source_checkpoint).resolve()
    target_checkpoint = Path(target_checkpoint).resolve()
    video_dir = Path(video_dir).resolve()
    commit = verify_sam2_checkout(sam2_repo)
    for label, checkpoint in (("source", source_checkpoint), ("target", target_checkpoint)):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint}")
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    started_at = _start_resource_measurement(device)
    prompt = load_binary_prompt(prompt_mask, object_id)
    _seed_everything(seed)
    source_predictor = build_sam2_video_predictor(
        config_file=source_config_file,
        ckpt_path=str(source_checkpoint),
        device=device,
    )
    source_state = source_predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    if not 0 <= switch_frame < int(source_state["num_frames"]) - 1:
        raise ValueError("switch_frame must leave at least one continuation frame")
    source_canonical, source_prefix_masks = _collect_prefix_reference(
        source_predictor,
        source_state,
        mask=prompt,
        object_id=object_id,
        switch_frame=switch_frame,
    )
    num_frames = int(source_state["num_frames"])
    del source_state, source_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _seed_everything(seed)
    target_predictor = build_sam2_video_predictor(
        config_file=target_config_file,
        ckpt_path=str(target_checkpoint),
        device=device,
    )
    target_state = target_predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    target_canonical, target_oracle_future = _collect_native(
        target_predictor,
        target_state,
        mask=prompt,
        object_id=object_id,
        switch_frame=switch_frame,
    )
    del target_state, target_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metadata = {
        "source_model_id": source_model_id,
        "target_model_id": target_model_id,
        "upstream_commit": commit,
        "video_id": video_dir.name,
        "object_id": object_id,
        "switch_frame": switch_frame,
        "num_frames": num_frames,
        "seed": seed,
        "device": device,
        "path_policy": "runtime_paths_not_stored",
    }
    preparation_resources = _resource_measurement(started_at, device)
    metadata["preparation_resources"] = preparation_resources
    cache = write_case_cache(
        output,
        source_canonical=source_canonical,
        target_canonical=target_canonical,
        source_prefix_masks=source_prefix_masks,
        target_oracle_future_masks=target_oracle_future,
        metadata=metadata,
    )
    return {
        **metadata,
        "source_contract": source_canonical.contract_dict(),
        "target_contract": target_canonical.contract_dict(),
        "resources": preparation_resources,
        "cache": cache,
    }


def run_cached_translator_handoff(
    *,
    case_cache: str | Path,
    sam2_repo: str | Path,
    target_config_file: str,
    target_checkpoint: str | Path,
    target_model_id: str,
    video_dir: str | Path,
    annotation_dir: str | Path | None = None,
    device: str | None = None,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
    artifact_dir: str | Path | None = None,
    translator: Any | None = None,
    translator_name: str = "direct_copy",
    candidate_label: str = "Direct Copy",
) -> dict[str, Any]:
    """Run only target continuation from a checksummed prepared case cache."""

    device = resolve_device(device, allow_mps=False)
    payload = load_case_cache(case_cache)
    metadata = payload["metadata"]
    if metadata.get("target_model_id") != target_model_id:
        raise ValueError(
            f"case cache target {metadata.get('target_model_id')!r} differs from "
            f"requested {target_model_id!r}"
        )
    video_dir = Path(video_dir).resolve()
    if metadata.get("video_id") != video_dir.name:
        raise ValueError(
            f"case cache video {metadata.get('video_id')!r} differs from {video_dir.name!r}"
        )
    sam2_repo = Path(sam2_repo).resolve()
    target_checkpoint = Path(target_checkpoint).resolve()
    commit = verify_sam2_checkout(sam2_repo)
    if metadata.get("upstream_commit") != commit:
        raise ValueError("case cache and runtime SAM 2 commits differ")
    if not target_checkpoint.is_file():
        raise FileNotFoundError(f"target checkpoint not found: {target_checkpoint}")
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    source_canonical = payload["source_canonical"]
    target_canonical = payload["target_canonical"]
    oracle_future = payload["target_oracle_future_masks"]
    if translator is None:
        translator = DirectCopyTranslator(target_canonical.spec)
    target_spec = getattr(translator, "target_spec", None)
    if target_spec is not None and target_spec != target_canonical.spec:
        raise ValueError(
            f"translator target spec {target_spec} differs from cached target spec "
            f"{target_canonical.spec}"
        )
    started_at = _start_resource_measurement(device)
    translation_started_at = time.perf_counter()
    translated = translator.translate(source_canonical)
    translation_wall_time_seconds = time.perf_counter() - translation_started_at

    _seed_everything(seed)
    predictor = build_sam2_video_predictor(
        config_file=target_config_file,
        ckpt_path=str(target_checkpoint),
        device=device,
    )
    backbone_calls: list[tuple[int, ...]] = []
    original_forward_image = predictor.forward_image

    def counted_forward_image(image: torch.Tensor):
        backbone_calls.append(tuple(image.shape))
        return original_forward_image(image)

    predictor.forward_image = counted_forward_image
    inference_state = init_sam2_inference_state_without_warmup(
        predictor,
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    calls_before_injection = len(backbone_calls)
    injection = inject_sam2_canonical_state(
        translated,
        predictor=predictor,
        inference_state=inference_state,
    )
    calls_after_injection = len(backbone_calls)
    switch_frame = int(metadata["switch_frame"])
    start_frame = switch_frame + 1
    candidate_future: dict[int, torch.Tensor] = {}
    for frame_idx, _object_ids, masks in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=start_frame,
        max_frame_num_to_track=int(metadata["num_frames"]) - start_frame,
        reverse=False,
    ):
        candidate_future[int(frame_idx)] = masks.detach().cpu().float()
    downstream = _compare_future_masks(oracle_future, candidate_future)
    candidate_resources = _resource_measurement(started_at, device)
    state_alignment = evaluate_state(translated, target_canonical)
    report = {
        "source_model_id": metadata["source_model_id"],
        "target_model_id": target_model_id,
        "translator": translator_name,
        "translator_parameter_count": int(translator.parameter_count()),
        "translation_wall_time_seconds": float(translation_wall_time_seconds),
        "upstream_commit": commit,
        "video_id": video_dir.name,
        "object_id": int(metadata["object_id"]),
        "switch_frame": switch_frame,
        "case_cache_sha256": Path(case_cache).with_suffix(
            Path(case_cache).suffix + ".sha256"
        ).read_text(encoding="ascii").split()[0],
        "state_alignment_to_target_native": state_alignment,
        "injection": injection,
        "backbone_calls_before_injection": calls_before_injection,
        "backbone_calls_during_injection": calls_after_injection - calls_before_injection,
        "backbone_calls_during_future_continuation": len(backbone_calls) - calls_after_injection,
        "mask_comparison_to_target_native": downstream,
        "seed": seed,
        "device": device,
        # Keep the generic key for the existing report renderer while naming the
        # scope explicitly for machine-readable comparisons.
        "resources": candidate_resources,
        "resources_candidate_only": candidate_resources,
        "resources_shared_reference_preparation": metadata.get("preparation_resources"),
    }
    if artifact_dir is not None:
        if annotation_dir is None:
            raise ValueError("annotation_dir is required when artifact_dir is set")
        report["artifacts"] = write_handoff_artifacts(
            video_dir=video_dir,
            annotation_dir=annotation_dir,
            object_id=int(metadata["object_id"]),
            oracle_masks=oracle_future,
            candidate_masks=candidate_future,
            output_dir=artifact_dir,
            report=report,
            candidate_label=candidate_label,
        )
    del inference_state, predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def run_cached_baseline(
    *,
    baseline: str,
    case_cache: str | Path,
    sam2_repo: str | Path,
    target_config_file: str,
    target_checkpoint: str | Path,
    target_model_id: str,
    video_dir: str | Path,
    prompt_mask: str | Path | None = None,
    annotation_dir: str | Path | None = None,
    replay_frames: int | None = None,
    device: str | None = None,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a non-translator target baseline from a prepared case cache.

    The cache supplies the source predictions used as prompts and the target-native
    future masks used for comparison.  Candidate target inference is always rerun,
    so latency and backbone-call counts remain method-specific.
    """

    device = resolve_device(device, allow_mps=False)
    cache_path = Path(case_cache).resolve()
    payload = load_case_cache(cache_path)
    metadata = payload["metadata"]
    if metadata.get("target_model_id") != target_model_id:
        raise ValueError(
            f"case cache target {metadata.get('target_model_id')!r} differs from "
            f"requested {target_model_id!r}"
        )
    video_dir = Path(video_dir).resolve()
    if metadata.get("video_id") != video_dir.name:
        raise ValueError(
            f"case cache video {metadata.get('video_id')!r} differs from "
            f"{video_dir.name!r}"
        )
    sam2_repo = Path(sam2_repo).resolve()
    target_checkpoint = Path(target_checkpoint).resolve()
    commit = verify_sam2_checkout(sam2_repo)
    if metadata.get("upstream_commit") != commit:
        raise ValueError("case cache and runtime SAM 2 commits differ")
    if not target_checkpoint.is_file():
        raise FileNotFoundError(f"target checkpoint not found: {target_checkpoint}")
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    switch_frame = int(metadata["switch_frame"])
    num_frames = int(metadata["num_frames"])
    object_id = int(metadata["object_id"])
    plan = plan_cached_baseline(
        baseline,
        switch_frame=switch_frame,
        replay_frames=replay_frames,
    )
    if baseline == "full_replay" and prompt_mask is None:
        raise ValueError("prompt_mask is required for full_replay")

    started_at = _start_resource_measurement(device)
    _seed_everything(seed)
    predictor = build_sam2_video_predictor(
        config_file=target_config_file,
        ckpt_path=str(target_checkpoint),
        device=device,
    )
    backbone_calls: list[tuple[int, ...]] = []
    original_forward_image = predictor.forward_image

    def counted_forward_image(image: torch.Tensor):
        backbone_calls.append(tuple(image.shape))
        return original_forward_image(image)

    predictor.forward_image = counted_forward_image
    inference_state = init_sam2_inference_state_without_warmup(
        predictor,
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    if int(inference_state["num_frames"]) != num_frames:
        raise ValueError(
            f"runtime video has {inference_state['num_frames']} frames but cache "
            f"expects {num_frames}"
        )
    prompt_frame = int(plan["prompt_frame"])
    if baseline == "target_reset":
        prompt = np.zeros(
            (int(inference_state["video_height"]), int(inference_state["video_width"])),
            dtype=bool,
        )
    elif baseline == "full_replay":
        prompt = load_binary_prompt(Path(prompt_mask), object_id)
    else:
        source_mask = payload["source_prefix_masks"].get(prompt_frame)
        if not isinstance(source_mask, torch.Tensor):
            raise ValueError(f"source prediction is missing for frame {prompt_frame}")
        prompt = binary_prompt_from_mask_logits(
            source_mask,
            height=int(inference_state["video_height"]),
            width=int(inference_state["video_width"]),
        )

    calls_before_prompt = len(backbone_calls)
    predictor.add_new_mask(
        inference_state,
        frame_idx=prompt_frame,
        obj_id=object_id,
        mask=prompt,
    )
    calls_after_prompt = len(backbone_calls)
    candidate_future: dict[int, torch.Tensor] = {}
    calls_after_switch: int | None = None
    for frame_idx, _object_ids, masks in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=prompt_frame,
        max_frame_num_to_track=num_frames - prompt_frame,
        reverse=False,
    ):
        frame = int(frame_idx)
        if frame == switch_frame:
            calls_after_switch = len(backbone_calls)
        if frame > switch_frame:
            candidate_future[frame] = masks.detach().cpu().float()
    if calls_after_switch is None:
        raise RuntimeError("baseline propagation did not reach switch_frame")

    oracle_future = payload["target_oracle_future_masks"]
    downstream = _compare_future_masks(oracle_future, candidate_future)
    resources = _resource_measurement(started_at, device)
    labels = {
        "target_reset": "Target Reset (blank object slot)",
        "last_mask": "Last-Mask",
        "replay_k": f"Replay-{plan['history_frames_reprocessed']}",
        "full_replay": "Full Replay / Target-native",
    }
    report = {
        "schema_version": "cmmt.cached_sam2_baseline.v1",
        "source_model_id": metadata["source_model_id"],
        "target_model_id": target_model_id,
        "translator": baseline,
        "baseline": baseline,
        "baseline_label": labels[baseline],
        "baseline_plan": plan,
        "target_reset_definition": (
            "SAM 2 cannot propagate an unregistered object. This proxy registers "
            "the object with an all-zero mask on the switch frame and transfers no "
            "source appearance, mask, pointer, or temporal memory."
            if baseline == "target_reset"
            else None
        ),
        "upstream_commit": commit,
        "video_id": video_dir.name,
        "object_id": object_id,
        "switch_frame": switch_frame,
        "case_cache_sha256": cache_path.with_suffix(
            cache_path.suffix + ".sha256"
        ).read_text(encoding="ascii").split()[0],
        "future_frames": sorted(candidate_future),
        "backbone_calls_before_prompt": calls_before_prompt,
        "backbone_calls_during_prompt": calls_after_prompt - calls_before_prompt,
        "backbone_calls_during_history_through_switch": (
            calls_after_switch - calls_after_prompt
        ),
        "backbone_calls_before_or_at_switch_total": calls_after_switch,
        "backbone_calls_during_future_continuation": (
            len(backbone_calls) - calls_after_switch
        ),
        "mask_comparison_to_target_native": downstream,
        "seed": seed,
        "device": device,
        "resources": resources,
        "resources_candidate_only": resources,
        "resources_shared_reference_preparation": metadata.get(
            "preparation_resources"
        ),
    }
    if artifact_dir is not None:
        if annotation_dir is None:
            raise ValueError("annotation_dir is required when artifact_dir is set")
        report["artifacts"] = write_handoff_artifacts(
            video_dir=video_dir,
            annotation_dir=annotation_dir,
            object_id=object_id,
            oracle_masks=oracle_future,
            candidate_masks=candidate_future,
            output_dir=artifact_dir,
            report=report,
            candidate_label=labels[baseline],
        )
    del inference_state, predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def _compare_future_masks(
    native: dict[int, torch.Tensor],
    injected: dict[int, torch.Tensor],
) -> dict[str, Any]:
    if native.keys() != injected.keys():
        raise ValueError(
            f"continuation frames differ: native={sorted(native)}, "
            f"injected={sorted(injected)}"
        )
    rows: list[dict[str, Any]] = []
    for frame in native:
        reference = native[frame]
        prediction = injected[frame]
        if reference.shape != prediction.shape:
            raise ValueError(
                f"frame {frame} mask shapes differ: {reference.shape} vs "
                f"{prediction.shape}"
            )
        difference = prediction - reference
        reference_binary = reference > 0
        prediction_binary = prediction > 0
        intersection = (reference_binary & prediction_binary).sum().item()
        union = (reference_binary | prediction_binary).sum().item()
        rows.append(
            {
                "frame": frame,
                "mse": float(difference.square().mean()),
                "max_abs_error": float(difference.abs().max()),
                "binary_iou": 1.0 if union == 0 else float(intersection / union),
            }
        )
    return {
        "frames": rows,
        "mean_mse": float(sum(row["mse"] for row in rows) / max(len(rows), 1)),
        "max_abs_error": float(max((row["max_abs_error"] for row in rows), default=0.0)),
        "mean_binary_iou": float(
            sum(row["binary_iou"] for row in rows) / max(len(rows), 1)
        ),
    }


def run_same_checkpoint_roundtrip(
    *,
    sam2_repo: str | Path,
    config_file: str,
    checkpoint: str | Path,
    model_id: str,
    video_dir: str | Path,
    prompt_mask: str | Path,
    object_id: int,
    switch_frame: int,
    device: str | None = None,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
) -> dict[str, Any]:
    """Compare native continuation with export→inject continuation."""

    device = resolve_device(device, allow_mps=False)
    sam2_repo = Path(sam2_repo).resolve()
    checkpoint = Path(checkpoint).resolve()
    video_dir = Path(video_dir).resolve()
    commit = verify_sam2_checkout(sam2_repo)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    started_at = _start_resource_measurement(device)
    _seed_everything(seed)
    mask = load_binary_prompt(prompt_mask, object_id)
    native_predictor = build_sam2_video_predictor(
        config_file=config_file,
        ckpt_path=str(checkpoint),
        device=device,
    )
    native_state = native_predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    if not 0 <= switch_frame < int(native_state["num_frames"]) - 1:
        raise ValueError("switch_frame must leave at least one continuation frame")
    canonical, native_future = _collect_native(
        native_predictor,
        native_state,
        mask=mask,
        object_id=object_id,
        switch_frame=switch_frame,
    )
    num_frames = int(native_state["num_frames"])
    del native_state, native_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _seed_everything(seed)
    injected_predictor = build_sam2_video_predictor(
        config_file=config_file,
        ckpt_path=str(checkpoint),
        device=device,
    )
    backbone_calls: list[tuple[int, ...]] = []
    original_forward_image = injected_predictor.forward_image

    def counted_forward_image(image: torch.Tensor):
        backbone_calls.append(tuple(image.shape))
        return original_forward_image(image)

    injected_predictor.forward_image = counted_forward_image
    injected_state = init_sam2_inference_state_without_warmup(
        injected_predictor,
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    calls_before_injection = len(backbone_calls)
    injection = inject_sam2_canonical_state(
        canonical,
        predictor=injected_predictor,
        inference_state=injected_state,
    )
    calls_after_injection = len(backbone_calls)
    injected_future: dict[int, torch.Tensor] = {}
    start_frame = switch_frame + 1
    for frame_idx, _object_ids, masks in injected_predictor.propagate_in_video(
        injected_state,
        start_frame_idx=start_frame,
        max_frame_num_to_track=num_frames - start_frame,
        reverse=False,
    ):
        injected_future[int(frame_idx)] = masks.detach().cpu().float()
    comparison = _compare_future_masks(native_future, injected_future)
    report = {
        "model_id": model_id,
        "upstream_commit": commit,
        "video_id": video_dir.name,
        "switch_frame": switch_frame,
        "future_frames": sorted(injected_future),
        "injection": injection,
        "backbone_calls_before_injection": calls_before_injection,
        "backbone_calls_during_injection": calls_after_injection
        - calls_before_injection,
        "backbone_calls_during_future_continuation": len(backbone_calls)
        - calls_after_injection,
        "comparison": comparison,
        "seed": seed,
        "device": device,
        "resources": _resource_measurement(started_at, device),
    }
    del injected_state, injected_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def run_same_checkpoint_prompt_timeline_roundtrip(
    *,
    sam2_repo: str | Path,
    config_file: str,
    checkpoint: str | Path,
    model_id: str,
    video_dir: str | Path,
    prompt_events: list[MaskPromptEvent],
    switch_frame: int,
    device: str | None = None,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
) -> dict[str, Any]:
    """Validate export→inject continuation for multi-object prompt timelines."""

    device = resolve_device(device, allow_mps=False)
    sam2_repo = Path(sam2_repo).resolve()
    checkpoint = Path(checkpoint).resolve()
    video_dir = Path(video_dir).resolve()
    normalized_events = [
        MaskPromptEvent(
            frame_index=int(event.frame_index),
            object_id=int(event.object_id),
            mask_path=Path(event.mask_path).resolve(),
        )
        for event in prompt_events
    ]
    commit = verify_sam2_checkout(sam2_repo)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    started_at = _start_resource_measurement(device)
    _seed_everything(seed)
    native_predictor = build_sam2_video_predictor(
        config_file=config_file,
        ckpt_path=str(checkpoint),
        device=device,
    )
    native_state = native_predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    if not 0 <= switch_frame < int(native_state["num_frames"]) - 1:
        raise ValueError("switch_frame must leave at least one continuation frame")
    canonical, native_future = _collect_native_prompt_timeline(
        native_predictor,
        native_state,
        prompt_events=normalized_events,
        switch_frame=switch_frame,
    )
    num_frames = int(native_state["num_frames"])
    del native_state, native_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _seed_everything(seed)
    injected_predictor = build_sam2_video_predictor(
        config_file=config_file,
        ckpt_path=str(checkpoint),
        device=device,
    )
    backbone_calls: list[tuple[int, ...]] = []
    original_forward_image = injected_predictor.forward_image

    def counted_forward_image(image: torch.Tensor):
        backbone_calls.append(tuple(image.shape))
        return original_forward_image(image)

    injected_predictor.forward_image = counted_forward_image
    injected_state = init_sam2_inference_state_without_warmup(
        injected_predictor,
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    calls_before_injection = len(backbone_calls)
    injection = inject_sam2_canonical_state(
        canonical,
        predictor=injected_predictor,
        inference_state=injected_state,
    )
    calls_after_injection = len(backbone_calls)
    injected_future: dict[int, torch.Tensor] = {}
    start_frame = switch_frame + 1
    for frame_idx, _object_ids, masks in injected_predictor.propagate_in_video(
        injected_state,
        start_frame_idx=start_frame,
        max_frame_num_to_track=num_frames - start_frame,
        reverse=False,
    ):
        injected_future[int(frame_idx)] = masks.detach().cpu().float()
    comparison = _compare_future_masks(native_future, injected_future)
    report = {
        "schema_version": "cmmt.sam2_prompt_timeline_roundtrip.v1",
        "model_id": model_id,
        "upstream_commit": commit,
        "video_id": video_dir.name,
        "switch_frame": switch_frame,
        "prompt_events": [
            {
                "frame_index": event.frame_index,
                "object_id": event.object_id,
                "mask_file": event.mask_path.name,
            }
            for event in normalized_events
        ],
        "future_frames": sorted(injected_future),
        "injection": injection,
        "backbone_calls_before_injection": calls_before_injection,
        "backbone_calls_during_injection": calls_after_injection
        - calls_before_injection,
        "backbone_calls_during_future_continuation": len(backbone_calls)
        - calls_after_injection,
        "comparison": comparison,
        "seed": seed,
        "device": device,
        "resources": _resource_measurement(started_at, device),
    }
    del injected_state, injected_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def run_same_checkpoint_correction_roundtrip(
    *,
    sam2_repo: str | Path,
    config_file: str,
    checkpoint: str | Path,
    model_id: str,
    video_dir: str | Path,
    prompt_events: list[MaskPromptEvent],
    correction_event: MaskPromptEvent,
    switch_frame: int,
    device: str | None = None,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
) -> dict[str, Any]:
    """Validate post-switch correction or the pre-switch replay fallback."""

    device = resolve_device(device, allow_mps=False)
    sam2_repo = Path(sam2_repo).resolve()
    checkpoint = Path(checkpoint).resolve()
    video_dir = Path(video_dir).resolve()
    initial_events = [
        MaskPromptEvent(int(event.frame_index), int(event.object_id), Path(event.mask_path).resolve())
        for event in prompt_events
    ]
    correction = MaskPromptEvent(
        int(correction_event.frame_index),
        int(correction_event.object_id),
        Path(correction_event.mask_path).resolve(),
    )
    commit = verify_sam2_checkout(sam2_repo)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    started_at = _start_resource_measurement(device)
    _seed_everything(seed)
    reference_predictor = build_sam2_video_predictor(
        config_file=config_file,
        ckpt_path=str(checkpoint),
        device=device,
    )
    reference_state = reference_predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    num_frames = int(reference_state["num_frames"])
    if not 0 <= switch_frame < num_frames - 1:
        raise ValueError("switch_frame must leave at least one continuation frame")
    _validate_prompt_events(
        initial_events, switch_frame=switch_frame, num_frames=num_frames
    )
    if correction.frame_index >= num_frames:
        raise ValueError("correction frame exceeds video length")
    route = plan_correction_route(
        correction_frame=correction.frame_index,
        switch_frame=switch_frame,
    )
    all_events = initial_events + [correction]
    reference_masks = _run_mask_prompt_timeline(
        reference_predictor,
        reference_state,
        prompt_events=all_events,
        final_frame=num_frames - 1,
    )
    del reference_state, reference_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    candidate_masks: dict[int, torch.Tensor] = {}
    injection: dict[str, Any] | None = None
    calls_before_injection = 0
    calls_during_injection = 0
    replay_start_frame: int | None = None
    history_frames_reprocessed = 0

    if route["requires_history_replay"]:
        replay_start_frame = min(event.frame_index for event in all_events)
        history_frames_reprocessed = switch_frame - replay_start_frame + 1
        _seed_everything(seed)
        candidate_predictor = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=str(checkpoint),
            device=device,
        )
        candidate_state = candidate_predictor.init_state(
            video_path=str(video_dir),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        candidate_masks = _run_mask_prompt_timeline(
            candidate_predictor,
            candidate_state,
            prompt_events=all_events,
            final_frame=num_frames - 1,
        )
    else:
        _seed_everything(seed)
        prefix_predictor = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=str(checkpoint),
            device=device,
        )
        prefix_state = prefix_predictor.init_state(
            video_path=str(video_dir),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        _run_mask_prompt_timeline(
            prefix_predictor,
            prefix_state,
            prompt_events=initial_events,
            final_frame=switch_frame,
        )
        canonical = canonicalize_sam2_inference_state(
            prefix_state,
            switch_frame=switch_frame,
            strict=True,
        )
        del prefix_state, prefix_predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _seed_everything(seed)
        candidate_predictor = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=str(checkpoint),
            device=device,
        )
        backbone_calls: list[tuple[int, ...]] = []
        original_forward_image = candidate_predictor.forward_image

        def counted_forward_image(image: torch.Tensor):
            backbone_calls.append(tuple(image.shape))
            return original_forward_image(image)

        candidate_predictor.forward_image = counted_forward_image
        candidate_state = init_sam2_inference_state_without_warmup(
            candidate_predictor,
            video_path=str(video_dir),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        calls_before_injection = len(backbone_calls)
        injection = inject_sam2_canonical_state(
            canonical,
            predictor=candidate_predictor,
            inference_state=candidate_state,
        )
        calls_during_injection = len(backbone_calls) - calls_before_injection
        if switch_frame + 1 < correction.frame_index:
            for frame_idx, _object_ids, masks in candidate_predictor.propagate_in_video(
                candidate_state,
                start_frame_idx=switch_frame + 1,
                max_frame_num_to_track=correction.frame_index - switch_frame - 1,
                reverse=False,
            ):
                candidate_masks[int(frame_idx)] = masks.detach().cpu().float()
        _frame_idx, _object_ids, corrected_masks = candidate_predictor.add_new_mask(
            candidate_state,
            frame_idx=correction.frame_index,
            obj_id=correction.object_id,
            mask=load_binary_prompt(correction.mask_path, correction.object_id),
        )
        candidate_masks[correction.frame_index] = corrected_masks.detach().cpu().float()
        for frame_idx, _object_ids, masks in candidate_predictor.propagate_in_video(
            candidate_state,
            start_frame_idx=correction.frame_index,
            max_frame_num_to_track=num_frames - correction.frame_index,
            reverse=False,
        ):
            candidate_masks[int(frame_idx)] = masks.detach().cpu().float()

    comparison_start = (
        switch_frame + 1 if route["requires_history_replay"] else correction.frame_index
    )
    reference_comparison = {
        frame: mask for frame, mask in reference_masks.items() if frame >= comparison_start
    }
    candidate_comparison = {
        frame: mask for frame, mask in candidate_masks.items() if frame >= comparison_start
    }
    comparison = _compare_future_masks(reference_comparison, candidate_comparison)
    report = {
        "schema_version": "cmmt.sam2_correction_roundtrip.v1",
        "model_id": model_id,
        "upstream_commit": commit,
        "video_id": video_dir.name,
        "switch_frame": switch_frame,
        "initial_prompt_events": [
            {
                "frame_index": event.frame_index,
                "object_id": event.object_id,
                "mask_file": event.mask_path.name,
            }
            for event in initial_events
        ],
        "correction_event": {
            "frame_index": correction.frame_index,
            "object_id": correction.object_id,
            "mask_file": correction.mask_path.name,
        },
        "route": route,
        "replay_start_frame": replay_start_frame,
        "history_frames_reprocessed": history_frames_reprocessed,
        "injection": injection,
        "backbone_calls_before_injection": calls_before_injection,
        "backbone_calls_during_injection": calls_during_injection,
        "comparison_start_frame": comparison_start,
        "comparison": comparison,
        "seed": seed,
        "device": device,
        "resources": _resource_measurement(started_at, device),
    }
    del candidate_state, candidate_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def run_same_checkpoint_repeated_switch_roundtrip(
    *,
    sam2_repo: str | Path,
    config_file: str,
    checkpoint: str | Path,
    model_id: str,
    video_dir: str | Path,
    prompt_events: list[MaskPromptEvent],
    switch_frames: tuple[int, int],
    device: str | None = None,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
) -> dict[str, Any]:
    """Validate two consecutive export→inject handoffs against a native run."""

    device = resolve_device(device, allow_mps=False)
    first_switch, second_switch = (int(value) for value in switch_frames)
    if not 0 <= first_switch < second_switch:
        raise ValueError("switch_frames must be strictly increasing and non-negative")
    sam2_repo = Path(sam2_repo).resolve()
    checkpoint = Path(checkpoint).resolve()
    video_dir = Path(video_dir).resolve()
    events = [
        MaskPromptEvent(int(event.frame_index), int(event.object_id), Path(event.mask_path).resolve())
        for event in prompt_events
    ]
    commit = verify_sam2_checkout(sam2_repo)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    started_at = _start_resource_measurement(device)
    _seed_everything(seed)
    native_predictor = build_sam2_video_predictor(
        config_file=config_file, ckpt_path=str(checkpoint), device=device
    )
    native_state = native_predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    num_frames = int(native_state["num_frames"])
    if second_switch >= num_frames - 1:
        raise ValueError("second switch must leave at least one continuation frame")
    _validate_prompt_events(events, switch_frame=first_switch, num_frames=num_frames)
    native_masks = _run_mask_prompt_timeline(
        native_predictor,
        native_state,
        prompt_events=events,
        final_frame=num_frames - 1,
    )
    del native_state, native_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _seed_everything(seed)
    prefix_predictor = build_sam2_video_predictor(
        config_file=config_file, ckpt_path=str(checkpoint), device=device
    )
    prefix_state = prefix_predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    _run_mask_prompt_timeline(
        prefix_predictor,
        prefix_state,
        prompt_events=events,
        final_frame=first_switch,
    )
    first_canonical = canonicalize_sam2_inference_state(
        prefix_state, switch_frame=first_switch, strict=True
    )
    del prefix_state, prefix_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    candidate_masks: dict[int, torch.Tensor] = {}
    injection_reports: list[dict[str, Any]] = []
    injection_backbone_calls: list[int] = []

    _seed_everything(seed)
    middle_predictor = build_sam2_video_predictor(
        config_file=config_file, ckpt_path=str(checkpoint), device=device
    )
    middle_calls: list[tuple[int, ...]] = []
    middle_forward_image = middle_predictor.forward_image

    def counted_middle_forward_image(image: torch.Tensor):
        middle_calls.append(tuple(image.shape))
        return middle_forward_image(image)

    middle_predictor.forward_image = counted_middle_forward_image
    middle_state = init_sam2_inference_state_without_warmup(
        middle_predictor,
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    before = len(middle_calls)
    injection_reports.append(
        inject_sam2_canonical_state(
            first_canonical, predictor=middle_predictor, inference_state=middle_state
        )
    )
    injection_backbone_calls.append(len(middle_calls) - before)
    for frame_idx, _object_ids, masks in middle_predictor.propagate_in_video(
        middle_state,
        start_frame_idx=first_switch + 1,
        max_frame_num_to_track=second_switch - first_switch,
        reverse=False,
    ):
        candidate_masks[int(frame_idx)] = masks.detach().cpu().float()
    second_canonical = canonicalize_sam2_inference_state(
        middle_state, switch_frame=second_switch, strict=True
    )
    del middle_state, middle_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _seed_everything(seed)
    final_predictor = build_sam2_video_predictor(
        config_file=config_file, ckpt_path=str(checkpoint), device=device
    )
    final_calls: list[tuple[int, ...]] = []
    final_forward_image = final_predictor.forward_image

    def counted_final_forward_image(image: torch.Tensor):
        final_calls.append(tuple(image.shape))
        return final_forward_image(image)

    final_predictor.forward_image = counted_final_forward_image
    final_state = init_sam2_inference_state_without_warmup(
        final_predictor,
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    before = len(final_calls)
    injection_reports.append(
        inject_sam2_canonical_state(
            second_canonical, predictor=final_predictor, inference_state=final_state
        )
    )
    injection_backbone_calls.append(len(final_calls) - before)
    for frame_idx, _object_ids, masks in final_predictor.propagate_in_video(
        final_state,
        start_frame_idx=second_switch + 1,
        max_frame_num_to_track=num_frames - second_switch - 1,
        reverse=False,
    ):
        candidate_masks[int(frame_idx)] = masks.detach().cpu().float()

    reference_after_first = {
        frame: mask for frame, mask in native_masks.items() if frame > first_switch
    }
    candidate_after_first = {
        frame: mask for frame, mask in candidate_masks.items() if frame > first_switch
    }
    reference_after_second = {
        frame: mask for frame, mask in native_masks.items() if frame > second_switch
    }
    candidate_after_second = {
        frame: mask for frame, mask in candidate_masks.items() if frame > second_switch
    }
    report = {
        "schema_version": "cmmt.sam2_repeated_switch_roundtrip.v1",
        "model_id": model_id,
        "upstream_commit": commit,
        "video_id": video_dir.name,
        "switch_frames": [first_switch, second_switch],
        "prompt_events": [
            {
                "frame_index": event.frame_index,
                "object_id": event.object_id,
                "mask_file": event.mask_path.name,
            }
            for event in events
        ],
        "injections": injection_reports,
        "backbone_calls_during_injections": injection_backbone_calls,
        "comparison_after_first_switch": _compare_future_masks(
            reference_after_first, candidate_after_first
        ),
        "comparison_after_second_switch": _compare_future_masks(
            reference_after_second, candidate_after_second
        ),
        "seed": seed,
        "device": device,
        "resources": _resource_measurement(started_at, device),
    }
    del final_state, final_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def run_cross_model_translator_handoff(
    *,
    sam2_repo: str | Path,
    source_config_file: str,
    source_checkpoint: str | Path,
    source_model_id: str,
    target_config_file: str,
    target_checkpoint: str | Path,
    target_model_id: str,
    video_dir: str | Path,
    prompt_mask: str | Path,
    object_id: int,
    switch_frame: int,
    device: str | None = None,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
    artifact_dir: str | Path | None = None,
    translator: Any | None = None,
    translator_name: str = "direct_copy",
    candidate_label: str = "Direct Copy",
) -> dict[str, Any]:
    """Run an end-to-end cross-model handoff with a supplied translator."""

    device = resolve_device(device, allow_mps=False)
    sam2_repo = Path(sam2_repo).resolve()
    source_checkpoint = Path(source_checkpoint).resolve()
    target_checkpoint = Path(target_checkpoint).resolve()
    video_dir = Path(video_dir).resolve()
    commit = verify_sam2_checkout(sam2_repo)
    for label, checkpoint in (
        ("source", source_checkpoint),
        ("target", target_checkpoint),
    ):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint}")
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    started_at = _start_resource_measurement(device)
    mask = load_binary_prompt(prompt_mask, object_id)
    _seed_everything(seed)
    source_predictor = build_sam2_video_predictor(
        config_file=source_config_file,
        ckpt_path=str(source_checkpoint),
        device=device,
    )
    source_state = source_predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    if not 0 <= switch_frame < int(source_state["num_frames"]) - 1:
        raise ValueError("switch_frame must leave at least one continuation frame")
    source_canonical = _collect_prefix(
        source_predictor,
        source_state,
        mask=mask,
        object_id=object_id,
        switch_frame=switch_frame,
    )
    num_frames = int(source_state["num_frames"])
    del source_state, source_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _seed_everything(seed)
    oracle_predictor = build_sam2_video_predictor(
        config_file=target_config_file,
        ckpt_path=str(target_checkpoint),
        device=device,
    )
    oracle_state = oracle_predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    target_canonical, oracle_future = _collect_native(
        oracle_predictor,
        oracle_state,
        mask=mask,
        object_id=object_id,
        switch_frame=switch_frame,
    )
    del oracle_state, oracle_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if translator is None:
        translator = DirectCopyTranslator(target_canonical.spec)
    target_spec = getattr(translator, "target_spec", None)
    if target_spec is not None and target_spec != target_canonical.spec:
        raise ValueError(
            f"translator target spec {target_spec} differs from runtime "
            f"target spec {target_canonical.spec}"
        )
    translated = translator.translate(source_canonical)
    state_alignment = evaluate_state(translated, target_canonical)

    _seed_everything(seed)
    target_predictor = build_sam2_video_predictor(
        config_file=target_config_file,
        ckpt_path=str(target_checkpoint),
        device=device,
    )
    backbone_calls: list[tuple[int, ...]] = []
    original_forward_image = target_predictor.forward_image

    def counted_forward_image(image: torch.Tensor):
        backbone_calls.append(tuple(image.shape))
        return original_forward_image(image)

    target_predictor.forward_image = counted_forward_image
    target_state = init_sam2_inference_state_without_warmup(
        target_predictor,
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    calls_before_injection = len(backbone_calls)
    injection = inject_sam2_canonical_state(
        translated,
        predictor=target_predictor,
        inference_state=target_state,
    )
    calls_after_injection = len(backbone_calls)
    candidate_future: dict[int, torch.Tensor] = {}
    start_frame = switch_frame + 1
    for frame_idx, _object_ids, masks in target_predictor.propagate_in_video(
        target_state,
        start_frame_idx=start_frame,
        max_frame_num_to_track=num_frames - start_frame,
        reverse=False,
    ):
        candidate_future[int(frame_idx)] = masks.detach().cpu().float()
    downstream = _compare_future_masks(oracle_future, candidate_future)
    report = {
        "source_model_id": source_model_id,
        "target_model_id": target_model_id,
        "translator": translator_name,
        "translator_parameter_count": int(translator.parameter_count()),
        "upstream_commit": commit,
        "video_id": video_dir.name,
        "switch_frame": switch_frame,
        "future_frames": sorted(candidate_future),
        "state_alignment_to_target_native": state_alignment,
        "injection": injection,
        "backbone_calls_before_injection": calls_before_injection,
        "backbone_calls_during_injection": calls_after_injection
        - calls_before_injection,
        "backbone_calls_during_future_continuation": len(backbone_calls)
        - calls_after_injection,
        "mask_comparison_to_target_native": downstream,
        "seed": seed,
        "device": device,
        "resources": _resource_measurement(started_at, device),
    }
    if artifact_dir is not None:
        report["artifacts"] = write_handoff_artifacts(
            video_dir=video_dir,
            annotation_dir=Path(prompt_mask).resolve().parent,
            object_id=object_id,
            oracle_masks=oracle_future,
            candidate_masks=candidate_future,
            output_dir=artifact_dir,
            report=report,
            candidate_label=candidate_label,
        )
    del target_state, target_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def run_cross_model_direct_handoff(**kwargs: Any) -> dict[str, Any]:
    """Backward-compatible entry point for the Direct Copy handoff baseline."""

    return run_cross_model_translator_handoff(**kwargs)
