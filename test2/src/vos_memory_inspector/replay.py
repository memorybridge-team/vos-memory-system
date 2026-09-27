"""Original-prompt plus bounded RGB replay for cached SAM 2 handoff cases.

This is deliberately separate from ``roundtrip.run_cached_baseline(replay_k)``.
The older baseline starts from a Source-predicted mask.  The protocol-frozen
baseline here always registers every object from its original user prompt and
then lets the Target process the last N pre-switch RGB frames in time order.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from .case_cache import load_case_cache
from .roundtrip import (
    _compare_future_masks,
    _resource_measurement,
    _seed_everything,
    _start_resource_measurement,
)
from .runner import load_binary_prompt
from .sam2_state import init_sam2_inference_state_without_warmup
from .upstream import verify_sam2_checkout


SCHEMA_VERSION = "cmmt.cached_original_prompt_replay.v1"


def _synchronize(device: str | torch.device) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def plan_original_prompt_replay(
    *,
    switch_frame: int,
    replay_frames: int,
    original_prompt_frame: int = 0,
) -> dict[str, Any]:
    """Resolve the exact anchor and recent-RGB frame set for Replay-N.

    ``replay_frames`` counts the contiguous recent window through and including
    ``switch_frame``.  This implementation requires the original prompt to
    precede that window; the current LVOS-10 protocol has one object prompted at
    frame 0, so this is fail-closed rather than silently mishandling late prompts.
    """

    if switch_frame < 0:
        raise ValueError("switch_frame must be non-negative")
    if replay_frames < 1:
        raise ValueError("replay_frames must be at least 1")
    if replay_frames > switch_frame + 1:
        raise ValueError(
            f"replay_frames={replay_frames} exceeds available prefix length "
            f"{switch_frame + 1}"
        )
    if not 0 <= original_prompt_frame <= switch_frame:
        raise ValueError("original_prompt_frame must be within the prefix")
    replay_start = switch_frame - replay_frames + 1
    if original_prompt_frame >= replay_start:
        raise ValueError(
            "this runner requires the original prompt before the recent replay window"
        )
    replay_window = list(range(replay_start, switch_frame + 1))
    unique_rgb_frames = sorted({original_prompt_frame, *replay_window})
    return {
        "method": f"original_prompt_replay_{replay_frames}",
        "original_prompt_frame": original_prompt_frame,
        "replay_frames": replay_frames,
        "replay_start_frame": replay_start,
        "replay_end_frame": switch_frame,
        "replay_window": replay_window,
        "unique_past_rgb_frames": unique_rgb_frames,
        "unique_past_rgb_frame_count": len(unique_rgb_frames),
        "uses_ground_truth_original_prompt": True,
        "uses_source_prediction_prompt": False,
    }


def _write_mask_set(
    masks: Mapping[int, torch.Tensor],
    output: Path,
    *,
    width: int,
    height: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("*.png"):
        old.unlink()
    for frame, logits in sorted(masks.items()):
        array = logits.detach().cpu().float().numpy().squeeze()
        if array.ndim != 2:
            raise ValueError(
                f"frame {frame} expected one-object 2D logits, got {array.shape}"
            )
        image = Image.fromarray((array > 0).astype(np.uint8) * 255)
        if image.size != (width, height):
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)
        image.save(output / f"{int(frame):05d}.png")


def run_cached_original_prompt_replay(
    *,
    case_cache: str | Path,
    sam2_repo: str | Path,
    target_config_file: str,
    target_checkpoint: str | Path,
    target_model_id: str,
    video_dir: str | Path,
    prompt_mask: str | Path,
    replay_frames: int,
    original_prompt_frame: int = 0,
    device: str = "cuda",
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run Target-native Original-Prompt(s)+Replay-N then continue after switch.

    The primary transition latency is ``timing.method_handoff_wall_time_seconds``:
    original-prompt registration plus synchronized processing of exactly N recent
    pre-switch frames.  Target model loading and video-state initialization are
    common costs and are recorded separately.  ``switch_ready_wall_time_seconds``
    includes all three for deployments that cannot preload the Target.
    """

    cache_path = Path(case_cache).resolve()
    payload = load_case_cache(cache_path)
    metadata = payload["metadata"]
    video_dir = Path(video_dir).resolve()
    sam2_repo = Path(sam2_repo).resolve()
    target_checkpoint = Path(target_checkpoint).resolve()
    prompt_mask = Path(prompt_mask).resolve()

    if metadata.get("target_model_id") != target_model_id:
        raise ValueError(
            f"case cache target {metadata.get('target_model_id')!r} differs from "
            f"requested {target_model_id!r}"
        )
    if metadata.get("video_id") != video_dir.name:
        raise ValueError(
            f"case cache video {metadata.get('video_id')!r} differs from "
            f"{video_dir.name!r}"
        )
    commit = verify_sam2_checkout(sam2_repo)
    if metadata.get("upstream_commit") != commit:
        raise ValueError("case cache and runtime SAM 2 commits differ")
    if not target_checkpoint.is_file():
        raise FileNotFoundError(f"target checkpoint not found: {target_checkpoint}")
    if not prompt_mask.is_file():
        raise FileNotFoundError(f"original prompt mask not found: {prompt_mask}")
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    switch_frame = int(metadata["switch_frame"])
    num_frames = int(metadata["num_frames"])
    object_id = int(metadata["object_id"])
    plan = plan_original_prompt_replay(
        switch_frame=switch_frame,
        replay_frames=replay_frames,
        original_prompt_frame=original_prompt_frame,
    )
    prompt = load_binary_prompt(prompt_mask, object_id)

    total_started_at = _start_resource_measurement(device)
    switch_ready_started_at = time.perf_counter()
    _seed_everything(seed)

    _synchronize(device)
    model_load_started_at = time.perf_counter()
    predictor = build_sam2_video_predictor(
        config_file=target_config_file,
        ckpt_path=str(target_checkpoint),
        device=device,
    )
    _synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_started_at

    backbone_calls: list[tuple[int, ...]] = []
    original_forward_image = predictor.forward_image

    def counted_forward_image(image: torch.Tensor):
        backbone_calls.append(tuple(image.shape))
        return original_forward_image(image)

    predictor.forward_image = counted_forward_image

    _synchronize(device)
    video_init_started_at = time.perf_counter()
    inference_state = init_sam2_inference_state_without_warmup(
        predictor,
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    _synchronize(device)
    video_init_seconds = time.perf_counter() - video_init_started_at
    if int(inference_state["num_frames"]) != num_frames:
        raise ValueError(
            f"runtime video has {inference_state['num_frames']} frames but cache "
            f"expects {num_frames}"
        )

    _synchronize(device)
    handoff_started_at = time.perf_counter()
    calls_before_prompt = len(backbone_calls)
    predictor.add_new_mask(
        inference_state,
        frame_idx=original_prompt_frame,
        obj_id=object_id,
        mask=prompt,
    )
    calls_after_prompt = len(backbone_calls)

    replayed_frames: list[int] = []
    for frame_idx, _object_ids, _masks in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=int(plan["replay_start_frame"]),
        max_frame_num_to_track=replay_frames - 1,
        reverse=False,
    ):
        replayed_frames.append(int(frame_idx))
    _synchronize(device)
    method_handoff_seconds = time.perf_counter() - handoff_started_at
    switch_ready_seconds = time.perf_counter() - switch_ready_started_at
    calls_after_replay = len(backbone_calls)

    expected_replay = list(plan["replay_window"])
    if replayed_frames != expected_replay:
        raise RuntimeError(
            f"replayed frames differ: expected={expected_replay}, got={replayed_frames}"
        )
    if calls_after_prompt - calls_before_prompt != 1:
        raise RuntimeError("original mask registration must encode exactly one RGB frame")
    if calls_after_replay - calls_after_prompt != replay_frames:
        raise RuntimeError(
            f"Replay-{replay_frames} encoded "
            f"{calls_after_replay - calls_after_prompt} recent frames"
        )

    candidate_future: dict[int, torch.Tensor] = {}
    future_start = switch_frame + 1
    _synchronize(device)
    future_started_at = time.perf_counter()
    for frame_idx, _object_ids, masks in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=future_start,
        max_frame_num_to_track=num_frames - future_start,
        reverse=False,
    ):
        candidate_future[int(frame_idx)] = masks.detach().cpu().float()
    _synchronize(device)
    future_seconds = time.perf_counter() - future_started_at

    oracle_future = payload["target_oracle_future_masks"]
    downstream = _compare_future_masks(oracle_future, candidate_future)
    resources = _resource_measurement(total_started_at, device)
    width = int(inference_state["video_width"])
    height = int(inference_state["video_height"])

    checksum = cache_path.with_suffix(cache_path.suffix + ".sha256").read_text(
        encoding="ascii"
    ).split()[0]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": str(plan["method"]),
        "baseline_label": f"Original-Prompt(s) + Replay-{replay_frames}",
        "source_model_id": metadata["source_model_id"],
        "target_model_id": target_model_id,
        "upstream_commit": commit,
        "video_id": video_dir.name,
        "object_id": object_id,
        "switch_frame": switch_frame,
        "case_cache_sha256": checksum,
        "baseline_plan": plan,
        "replayed_frames": replayed_frames,
        "future_frames": sorted(candidate_future),
        "backbone_calls_during_original_prompt": calls_after_prompt - calls_before_prompt,
        "backbone_calls_during_recent_replay": calls_after_replay - calls_after_prompt,
        "backbone_calls_before_or_at_switch_total": calls_after_replay,
        "backbone_calls_during_future_continuation": len(backbone_calls) - calls_after_replay,
        "state_transfer_bytes": 0,
        "timing": {
            "scope": {
                "method_handoff": (
                    "original prompt registration plus synchronized Target processing "
                    "of the N recent RGB frames; excludes model load and video init"
                ),
                "switch_ready": (
                    "Target model load + video init + method handoff; excludes future "
                    "continuation"
                ),
            },
            "target_model_load_wall_time_seconds": float(model_load_seconds),
            "target_video_init_wall_time_seconds": float(video_init_seconds),
            "method_handoff_wall_time_seconds": float(method_handoff_seconds),
            "switch_ready_wall_time_seconds": float(switch_ready_seconds),
            "future_continuation_wall_time_seconds": float(future_seconds),
        },
        "mask_comparison_to_target_native": downstream,
        "seed": seed,
        "device": device,
        "resources": resources,
        "resources_candidate_only": resources,
        "resources_shared_reference_preparation": metadata.get("preparation_resources"),
    }

    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir).resolve()
        _write_mask_set(
            candidate_future,
            artifact_dir / "candidate_masks",
            width=width,
            height=height,
        )
        _write_mask_set(
            oracle_future,
            artifact_dir / "oracle_masks",
            width=width,
            height=height,
        )
        report["artifacts"] = {
            "candidate_masks": "candidate_masks/",
            "oracle_masks": "oracle_masks/",
            "comparisons": None,
        }
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "replay_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    del inference_state, predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report
