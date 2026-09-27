"""Single-pass aligned/native state-pair extraction for LVOS v2 and VOST."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .checkpoint_audit import sha256_file
from .pair_bank import PairBankWriter, SemanticRecordKey, slice_record
from .precondition_manifest import validate_train_split_manifest
from .sam2_state import canonicalize_sam2_inference_state, init_sam2_inference_state_without_warmup
from .state_schema import CanonicalState, validate_paired_state_contract
from .upstream import verify_sam2_checkout


def _mask_path(annotation_dir: Path, frame_id: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = annotation_dir / f"{frame_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"annotation frame {frame_id} is missing in {annotation_dir}")


def _object_mask(annotation_dir: Path, frame_id: str, object_id: int) -> np.ndarray:
    values = np.asarray(Image.open(_mask_path(annotation_dir, frame_id)))
    if values.ndim == 3:
        values = values[..., 0]
    return values == object_id


def _object_timelines(annotation_dir: Path, annotated_frames: list[str]) -> dict[int, list[str]]:
    timelines: dict[int, list[str]] = {}
    for frame_id in annotated_frames:
        values = np.asarray(Image.open(_mask_path(annotation_dir, frame_id)))
        if values.ndim == 3:
            values = values[..., 0]
        for object_id in np.unique(values):
            integer = int(object_id)
            if integer in {0, 255}:
                continue
            timelines.setdefault(integer, []).append(frame_id)
    return timelines


def _uniform(values: list[int], cap: int) -> list[int]:
    if len(values) <= cap:
        return list(values)
    positions = np.linspace(0, len(values) - 1, num=cap).round().astype(int)
    return [values[index] for index in sorted(set(positions.tolist()))]


def _switches(first: int, last: int) -> list[int]:
    if last <= first:
        return [first]
    return sorted({int(round(first + quantile * (last - first))) for quantile in (0.25, 0.5, 0.75)})


def _build_predictor(sam2_repo: Path, config: str, checkpoint: Path, device: str) -> Any:
    repo_text = str(sam2_repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(config_file=config, ckpt_path=str(checkpoint), device=device)
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    return predictor


def _state_by_frame(state: CanonicalState, object_slot: int = 0) -> dict[int, int]:
    result = {}
    for record_slot in range(state.spatial_memory.shape[2]):
        if bool(state.validity[0, object_slot, record_slot]):
            frame = int(state.frame_indices[0, object_slot, record_slot])
            if frame in result:
                raise ValueError(f"canonical state contains duplicate frame {frame}")
            result[frame] = record_slot
    return result


def _write_snapshot_records(
    writer: PairBankWriter,
    source: CanonicalState,
    target: CanonicalState,
    *,
    video_id: str,
    object_id: int,
    split: str,
    pair_type: str,
    dataset: str,
    release: str,
    frame_ids: list[str],
    switch_frame: int,
    training_frames: set[int] | None = None,
) -> list[str]:
    validate_paired_state_contract(source, target)
    source_slots = _state_by_frame(source)
    target_slots = _state_by_frame(target)
    if set(source_slots) != set(target_slots):
        raise ValueError("source/target history record frames differ")
    record_ids = []
    for frame in sorted(source_slots):
        source_record = slice_record(source, 0, source_slots[frame])
        target_record = slice_record(target, 0, target_slots[frame])
        source_record.switch_frame = switch_frame
        target_record.switch_frame = switch_frame
        is_conditioning = bool(source_record.is_conditioning[0, 0, 0])
        key = SemanticRecordKey(
            dataset=dataset,
            release=release,
            split=split,
            pair_type=pair_type,
            video_id=video_id,
            object_id=str(object_id),
            frame_idx=frame,
            is_conditioning=is_conditioning,
        )
        record_ids.append(
            writer.write_record(
                key,
                source_record,
                target_record,
                frame_id=frame_ids[frame],
                predictor_frame_idx=frame,
                training_eligible=(
                    True if training_frames is None else frame in training_frames
                ),
            )
        )
    writer.write_snapshot(
        video_id=video_id,
        object_id=str(object_id),
        switch_frame=switch_frame,
        record_ids=record_ids,
        frame_id_map={frame_id: index for index, frame_id in enumerate(frame_ids)},
    )
    return record_ids


@torch.inference_mode()
def _aligned_states(
    source_predictor: Any,
    target_predictor: Any,
    *,
    video_dir: Path,
    annotation_dir: Path,
    frame_ids: list[str],
    sampled_frames: list[int],
    object_id: int,
) -> tuple[CanonicalState, CanonicalState]:
    states = []
    for predictor in (source_predictor, target_predictor):
        inference = init_sam2_inference_state_without_warmup(
            predictor, video_path=str(video_dir), offload_video_to_cpu=True, offload_state_to_cpu=True
        )
        for frame in sampled_frames:
            predictor.add_new_mask(
                inference,
                frame_idx=frame,
                obj_id=object_id,
                mask=_object_mask(annotation_dir, frame_ids[frame], object_id),
            )
        predictor.propagate_in_video_preflight(inference)
        canonical = canonicalize_sam2_inference_state(
            inference, switch_frame=max(sampled_frames), strict=True
        )
        canonical.metadata.update({"pair_type": "aligned", "controlled_mask": "ground_truth"})
        states.append(canonical)
        del inference
    return states[0], states[1]


@torch.inference_mode()
def _native_snapshots(
    predictor: Any,
    *,
    video_dir: Path,
    annotation_dir: Path,
    frame_ids: list[str],
    object_id: int,
    prompt_frame: int,
    switch_frames: list[int],
) -> dict[int, CanonicalState]:
    inference = init_sam2_inference_state_without_warmup(
        predictor, video_path=str(video_dir), offload_video_to_cpu=True, offload_state_to_cpu=True
    )
    predictor.add_new_mask(
        inference,
        frame_idx=prompt_frame,
        obj_id=object_id,
        mask=_object_mask(annotation_dir, frame_ids[prompt_frame], object_id),
    )
    snapshots: dict[int, CanonicalState] = {}
    maximum = max(switch_frames)
    for frame, _object_ids, _masks in predictor.propagate_in_video(
        inference,
        start_frame_idx=prompt_frame,
        max_frame_num_to_track=maximum - prompt_frame,
        reverse=False,
    ):
        frame = int(frame)
        if frame in switch_frames:
            snapshot = canonicalize_sam2_inference_state(inference, switch_frame=frame, strict=True)
            snapshot.metadata.update({"pair_type": "native", "prompt_frame": prompt_frame})
            snapshots[frame] = snapshot
    del inference
    if set(snapshots) != set(switch_frames):
        raise RuntimeError(f"native propagation missed switch frames {sorted(set(switch_frames) - set(snapshots))}")
    return snapshots


def extract_pair_bank(
    *,
    manifest_path: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    sam2_repo: str | Path,
    source_config: str,
    source_checkpoint: str | Path,
    target_config: str,
    target_checkpoint: str | Path,
    pair_types: tuple[str, ...] = ("aligned", "native"),
    screening_only: bool = True,
    tiny_overfit_only: bool = False,
    device: str = "cuda",
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_train_split_manifest(manifest)
    dataset_root = Path(dataset_root).resolve()
    sam2_repo = Path(sam2_repo).resolve()
    source_checkpoint = Path(source_checkpoint).resolve()
    target_checkpoint = Path(target_checkpoint).resolve()
    commit = verify_sam2_checkout(sam2_repo)
    provenance = {
        "manifest_sha256": manifest["content_sha256"],
        "upstream_commit": commit,
        "source_config": source_config,
        "target_config": target_config,
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "target_checkpoint_sha256": sha256_file(target_checkpoint),
    }
    source_predictor = _build_predictor(sam2_repo, source_config, source_checkpoint, device)
    target_predictor = _build_predictor(sam2_repo, target_config, target_checkpoint, device)
    selected = (
        {manifest["screening"]["tiny_overfit"]["video_id"]}
        if tiny_overfit_only
        else {
            video
            for split in ("fit", "dev", "test")
            for video in manifest["screening"]["selected"][split]
        }
    )
    rows = [row for row in manifest["videos"] if not screening_only or row["video_id"] in selected]
    summary: dict[str, Any] = {"videos": {}, "failures": []}
    summary["selection"] = (
        "tiny_overfit" if tiny_overfit_only else "screening" if screening_only else "full_train"
    )
    for row in rows:
        video_id = str(row["video_id"])
        split = str(row["split"])
        video_dir = dataset_root / row["video_dir"]
        annotation_dir = dataset_root / row["annotation_dir"]
        frame_ids = list(row["frame_ids"])
        timelines = _object_timelines(annotation_dir, list(row["annotated_frame_ids"]))
        video_count = 0
        for object_id, present_ids in sorted(timelines.items()):
            present_frames = [row["frame_id_to_predictor_idx"][frame_id] for frame_id in present_ids]
            sampled = _uniform(present_frames, 16)
            if video_count + len(sampled) > 128:
                sampled = sampled[: max(0, 128 - video_count)]
            if not sampled:
                continue
            for pair_type in pair_types:
                writer = PairBankWriter(
                    output_root,
                    dataset=manifest["dataset"],
                    release=manifest["release"],
                    pair_type=pair_type,
                    split=split,
                    provenance=provenance,
                )
                try:
                    if pair_type == "aligned":
                        source, target = _aligned_states(
                            source_predictor,
                            target_predictor,
                            video_dir=video_dir,
                            annotation_dir=annotation_dir,
                            frame_ids=frame_ids,
                            sampled_frames=sampled,
                            object_id=object_id,
                        )
                        _write_snapshot_records(
                            writer,
                            source,
                            target,
                            video_id=video_id,
                            object_id=object_id,
                            split=split,
                            pair_type=pair_type,
                            dataset=manifest["dataset"],
                            release=manifest["release"],
                            frame_ids=frame_ids,
                            switch_frame=max(sampled),
                        )
                    elif pair_type == "native":
                        switches = _switches(min(present_frames), len(frame_ids) - 1)
                        native_training_frames = set(
                            _uniform(
                                [frame for frame in present_frames if frame <= max(switches)],
                                16,
                            )
                        )
                        source_snapshots = _native_snapshots(
                            source_predictor,
                            video_dir=video_dir,
                            annotation_dir=annotation_dir,
                            frame_ids=frame_ids,
                            object_id=object_id,
                            prompt_frame=min(present_frames),
                            switch_frames=switches,
                        )
                        target_snapshots = _native_snapshots(
                            target_predictor,
                            video_dir=video_dir,
                            annotation_dir=annotation_dir,
                            frame_ids=frame_ids,
                            object_id=object_id,
                            prompt_frame=min(present_frames),
                            switch_frames=switches,
                        )
                        for switch in switches:
                            _write_snapshot_records(
                                writer,
                                source_snapshots[switch],
                                target_snapshots[switch],
                                video_id=video_id,
                                object_id=object_id,
                                split=split,
                                pair_type=pair_type,
                                dataset=manifest["dataset"],
                                release=manifest["release"],
                                frame_ids=frame_ids,
                                switch_frame=switch,
                                training_frames=native_training_frames,
                            )
                    else:
                        raise ValueError(f"unsupported pair type: {pair_type}")
                except Exception as exc:
                    failure = {
                        "video_id": video_id,
                        "object_id": object_id,
                        "pair_type": pair_type,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    writer.log_failure(failure)
                    summary["failures"].append(failure)
            video_count += len(sampled)
            if video_count >= 128:
                break
        summary["videos"][video_id] = {"objects": len(timelines), "sampled_records": video_count}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary["schema_version"] = "cmmt.state_pair_extraction_report.v1"
    summary["provenance"] = provenance
    report_path = Path(output_root).resolve() / "extraction_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
