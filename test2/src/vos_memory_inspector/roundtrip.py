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


CACHED_BASELINES = (
    "empty_memory_reset",
    "target_reset",
    "last_mask",
    "replay_k",
    "full_replay",
)


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
    if baseline == "empty_memory_reset":
        return {
            "baseline": baseline,
            "prompt_frame": switch_frame + 1,
            "prompt_source": "none",
            "history_frames_reprocessed": 0,
            "uses_ground_truth_prompt": False,
            "uses_source_prediction_prompt": False,
            "diagnostic_limitation": (
                "object slot is registered without appearance or prompt evidence"
            ),
        }
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
    device: str = "cuda",
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
) -> dict[str, Any]:
    """Compute source prefix and target oracle once for reuse by all baselines."""

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
    device: str = "cuda",
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
    artifact_dir: str | Path | None = None,
    translator: Any | None = None,
    translator_name: str = "direct_copy",
    candidate_label: str = "Direct Copy",
) -> dict[str, Any]:
    """Run only target continuation from a checksummed prepared case cache."""

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
    device: str = "cuda",
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
    candidate_future: dict[int, torch.Tensor] = {}
    calls_after_switch: int | None = None
    if baseline == "empty_memory_reset":
        # Register object identity only. The first suffix frame runs without a
        # point/mask prompt or previous memory and seeds its own future memory.
        object_index = predictor._obj_id_to_idx(inference_state, object_id)
        object_output = inference_state["output_dict_per_obj"][object_index]
        calls_before_prompt = len(backbone_calls)
        current_out, low_res_masks = predictor._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=object_output,
            frame_idx=prompt_frame,
            batch_size=1,
            is_init_cond_frame=True,
            point_inputs=None,
            mask_inputs=None,
            reverse=False,
            run_mem_encoder=True,
        )
        object_output["cond_frame_outputs"][prompt_frame] = current_out
        inference_state["frames_tracked_per_obj"][object_index][prompt_frame] = {
            "reverse": False
        }
        calls_after_prompt = len(backbone_calls)
        _, video_res_masks = predictor._get_orig_video_res_output(
            inference_state, low_res_masks
        )
        candidate_future[prompt_frame] = video_res_masks.detach().cpu().float()
        calls_after_switch = calls_before_prompt
        if prompt_frame + 1 < num_frames:
            for frame_idx, _object_ids, masks in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=prompt_frame + 1,
                max_frame_num_to_track=num_frames - prompt_frame - 1,
                reverse=False,
            ):
                candidate_future[int(frame_idx)] = masks.detach().cpu().float()
    else:
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
        "empty_memory_reset": "Empty-memory Reset (unprompted diagnostic)",
        "target_reset": "Target Reset (legacy empty-mask prompt proxy)",
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
        "empty_memory_reset_definition": (
            "At switch+1 the object registry is created without a mask, point, "
            "source state, or replay. The first suffix frame runs through the "
            "target no-memory path and seeds its own later memory."
            if baseline == "empty_memory_reset"
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
    device: str = "cuda",
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
) -> dict[str, Any]:
    """Compare native continuation with export→inject continuation."""

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
    device: str = "cuda",
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
) -> dict[str, Any]:
    """Validate export→inject continuation for multi-object prompt timelines."""

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
    device: str = "cuda",
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
    artifact_dir: str | Path | None = None,
    translator: Any | None = None,
    translator_name: str = "direct_copy",
    candidate_label: str = "Direct Copy",
) -> dict[str, Any]:
    """Run an end-to-end cross-model handoff with a supplied translator."""

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
