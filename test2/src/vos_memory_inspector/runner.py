from __future__ import annotations

import gc
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .attention_hook import MemoryAttentionProbe
from .manifest import DumpPolicy, ManifestWriter
from .probe import ProbeConfig, StateProbe
from .sam2_state import canonicalize_sam2_inference_state
from .upstream import verify_sam2_checkout


def load_binary_prompt(path: str | Path, object_id: int) -> np.ndarray:
    labels = np.asarray(Image.open(path))
    if labels.ndim == 3:
        labels = labels[..., 0]
    mask = labels == object_id
    if not mask.any():
        unique = np.unique(labels).tolist()
        raise ValueError(
            f"Prompt mask {path} contains no pixels for object_id={object_id}; "
            f"labels={unique[:20]}"
        )
    return mask


def run_video_probe(
    *,
    sam2_repo: str | Path,
    config_file: str,
    checkpoint: str | Path,
    model_id: str,
    video_dir: str | Path,
    prompt_mask: str | Path,
    object_id: int,
    switch_frame: int,
    jsonl_path: str | Path,
    csv_path: str | Path | None = None,
    dump_dir: str | Path | None = None,
    dump_tensors: tuple[str, ...] = (),
    device: str = "cuda",
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    allow_upstream_mismatch: bool = False,
    seed: int = 0,
    canonical_state_path: str | Path | None = None,
) -> dict[str, Any]:
    sam2_repo = Path(sam2_repo).resolve()
    video_dir = Path(video_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    if switch_frame < 0:
        raise ValueError("switch_frame must be non-negative")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video frame directory not found: {video_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    upstream_commit = verify_sam2_checkout(
        sam2_repo, allow_mismatch=allow_upstream_mismatch
    )
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        config_file=config_file,
        ckpt_path=str(checkpoint),
        device=device,
    )
    inference_state = predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    if switch_frame >= int(inference_state["num_frames"]):
        raise ValueError(
            f"switch_frame={switch_frame} exceeds final frame "
            f"{int(inference_state['num_frames']) - 1}"
        )
    mask = load_binary_prompt(prompt_mask, object_id)
    config = ProbeConfig(
        model_id=model_id,
        upstream_commit=upstream_commit,
        video_id=video_dir.name,
        switch_frame=switch_frame,
    )
    dump_policy = DumpPolicy(
        None if dump_dir is None else Path(dump_dir), frozenset(dump_tensors)
    )
    frames_recorded: list[int] = []
    with ManifestWriter(jsonl_path, csv_path) as writer:
        state_probe = StateProbe(config, writer, dump_policy=dump_policy)
        attention_probe = MemoryAttentionProbe(
            predictor,
            inference_state,
            config,
            writer,
            dump_policy=dump_policy,
        )
        with attention_probe:
            predictor.add_new_mask(
                inference_state,
                frame_idx=0,
                obj_id=object_id,
                mask=mask,
            )
            for frame_idx, object_ids, _ in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=switch_frame,
                reverse=False,
            ):
                state_probe.record_frame(
                    inference_state,
                    int(frame_idx),
                    object_ids=list(object_ids),
                )
                frames_recorded.append(int(frame_idx))
                if frame_idx >= switch_frame:
                    break
    if canonical_state_path is not None:
        canonical_path = Path(canonical_state_path)
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical = canonicalize_sam2_inference_state(
            inference_state, switch_frame=switch_frame, strict=True
        )
        torch.save(canonical, canonical_path)
    summary = {
        "model_id": model_id,
        "upstream_commit": upstream_commit,
        "video_id": video_dir.name,
        "switch_frame": switch_frame,
        "frames_recorded": frames_recorded,
        "jsonl_path": str(Path(jsonl_path).resolve()),
        "csv_path": None if csv_path is None else str(Path(csv_path).resolve()),
        "dump_tensors": list(dump_tensors),
        "seed": seed,
        "canonical_state_path": (
            None
            if canonical_state_path is None
            else str(Path(canonical_state_path).resolve())
        ),
    }
    del inference_state, predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary

