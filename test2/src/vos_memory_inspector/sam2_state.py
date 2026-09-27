"""SAM 2 predictor-state probing and canonical handoff conversion."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from .state_inspector import InspectionReport, inspect_state
from .state_schema import CanonicalState


COND_KEY = "cond_frame_outputs"
NON_COND_KEY = "non_cond_frame_outputs"
SOURCE_COMPACT_OUTPUT_KEYS = (
    "maskmem_features",
    "maskmem_pos_enc",
    "pred_masks",
    "obj_ptr",
    "object_score_logits",
)


@dataclass(frozen=True)
class _SAM2Record:
    object_index: int
    object_id: Any
    frame_index: int
    is_conditioning: bool
    output: Mapping[str, Any]


def _object_id(inference_state: Mapping[str, Any], object_index: int) -> Any:
    reverse = inference_state.get("obj_idx_to_id", {})
    if object_index in reverse:
        return reverse[object_index]
    ids = inference_state.get("obj_ids", ())
    if object_index < len(ids):
        return ids[object_index]
    return object_index


def _collect_records(
    inference_state: Mapping[str, Any], switch_frame: int
) -> tuple[list[int], dict[int, list[_SAM2Record]]]:
    per_object = inference_state.get("output_dict_per_obj")
    if not isinstance(per_object, Mapping) or not per_object:
        raise ValueError("inference_state has no output_dict_per_obj records")
    object_indices = sorted(int(index) for index in per_object)
    result: dict[int, list[_SAM2Record]] = {}
    for object_index in object_indices:
        object_output = per_object[object_index]
        records: list[_SAM2Record] = []
        for storage_key, is_conditioning in ((COND_KEY, True), (NON_COND_KEY, False)):
            frame_outputs = object_output.get(storage_key, {})
            for raw_frame, output in frame_outputs.items():
                frame = int(raw_frame)
                if frame <= switch_frame:
                    records.append(
                        _SAM2Record(
                            object_index=object_index,
                            object_id=_object_id(inference_state, object_index),
                            frame_index=frame,
                            is_conditioning=is_conditioning,
                            output=output,
                        )
                    )
        records.sort(key=lambda record: (record.frame_index, not record.is_conditioning))
        if not records:
            raise ValueError(
                f"object {object_index} has no finalized output at or before frame {switch_frame}"
            )
        result[object_index] = records
    return object_indices, result


def canonicalize_sam2_inference_state(
    inference_state: Mapping[str, Any],
    *,
    switch_frame: int,
    strict: bool = True,
) -> CanonicalState:
    """Extract finalized per-object histories into ``CanonicalState``.

    The function does not assume specific model dimensions.  It discovers the first
    complete compact output and validates every later record against it.  Prompt
    Source masks are retained as diagnostic archive metadata. Video dimensions,
    prompt inputs and frame-tracking records are runtime/dataset-owned and are
    not part of the minimal handoff payload.
    """

    # Pinned SAM 2 offloads ``maskmem_features`` and ``pred_masks`` to CPU with
    # ``non_blocking=True``.  The most recently produced record can therefore still
    # be in flight when a caller exports state immediately after a propagation
    # yield.  Reading that host tensor before the copy completes silently captures
    # a partially written memory feature.  Synchronize the producer device once at
    # the export boundary so every copied record is a stable snapshot.
    compute_device = torch.device(inference_state.get("device", "cpu"))
    storage_device = torch.device(inference_state.get("storage_device", compute_device))
    if (
        compute_device.type == "cuda"
        and storage_device.type == "cpu"
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize(compute_device)

    object_indices, records_by_object = _collect_records(inference_state, switch_frame)
    complete = [
        record
        for records in records_by_object.values()
        for record in records
        if isinstance(record.output.get("maskmem_features"), torch.Tensor)
        and isinstance(record.output.get("obj_ptr"), torch.Tensor)
        and isinstance(record.output.get("object_score_logits"), torch.Tensor)
    ]
    if not complete:
        raise ValueError(
            "No complete SAM 2 compact output found. Run propagation preflight so "
            "conditioning records receive mask-memory features."
        )
    exemplar = complete[0]
    feature0 = exemplar.output["maskmem_features"]
    pointer0 = exemplar.output["obj_ptr"]
    presence0 = exemplar.output["object_score_logits"]
    if feature0.ndim != 4 or feature0.shape[0] != 1:
        raise ValueError(f"maskmem_features must be [1,C,H,W], got {feature0.shape}")
    if pointer0.ndim != 2 or pointer0.shape[0] != 1:
        raise ValueError(f"obj_ptr must be [1,D], got {pointer0.shape}")
    if presence0.shape != (1, 1):
        raise ValueError(
            f"object_score_logits must be [1,1], got {presence0.shape}"
        )

    _, channels, height, width = feature0.shape
    pointer_dim = pointer0.shape[-1]
    objects = len(object_indices)
    records = max(len(items) for items in records_by_object.values())
    spatial = feature0.new_zeros((1, objects, records, channels, height, width))
    pointer = pointer0.new_zeros((1, objects, records, pointer_dim))
    presence = presence0.new_zeros((1, objects, records, 1))
    frame_indices = torch.full((1, objects, records), -1, dtype=torch.long)
    slot_order = torch.full((1, objects, records), -1, dtype=torch.long)
    is_conditioning = torch.zeros((1, objects, records), dtype=torch.bool)
    validity = torch.zeros((1, objects, records), dtype=torch.bool)
    positional_records: dict[str, Any] = {}
    preserved_masks: dict[str, torch.Tensor] = {}

    for object_slot, object_index in enumerate(object_indices):
        for record_slot, record in enumerate(records_by_object[object_index]):
            output = record.output
            missing = [
                key
                for key in ("maskmem_features", "obj_ptr", "object_score_logits")
                if not isinstance(output.get(key), torch.Tensor)
            ]
            if missing:
                if strict:
                    raise ValueError(
                        f"incomplete SAM 2 output object={record.object_id} "
                        f"frame={record.frame_index}: missing {missing}"
                    )
                continue
            feature = output["maskmem_features"]
            obj_ptr = output["obj_ptr"]
            score = output["object_score_logits"]
            expected = {
                "maskmem_features": (1, channels, height, width),
                "obj_ptr": (1, pointer_dim),
                "object_score_logits": (1, 1),
            }
            actual = {
                "maskmem_features": tuple(feature.shape),
                "obj_ptr": tuple(obj_ptr.shape),
                "object_score_logits": tuple(score.shape),
            }
            if actual != expected:
                raise ValueError(
                    f"SAM 2 runtime shape changed within one state at object "
                    f"{record.object_id}, frame {record.frame_index}: {actual} vs {expected}"
                )
            spatial[0, object_slot, record_slot].copy_(feature[0].to(spatial.device))
            pointer[0, object_slot, record_slot].copy_(obj_ptr[0].to(pointer.device))
            presence[0, object_slot, record_slot].copy_(score[0].to(presence.device))
            frame_indices[0, object_slot, record_slot] = record.frame_index
            slot_order[0, object_slot, record_slot] = record_slot
            is_conditioning[0, object_slot, record_slot] = record.is_conditioning
            validity[0, object_slot, record_slot] = True
            record_key = f"object={object_slot}/record={record_slot}"
            if output.get("maskmem_pos_enc") is not None:
                positional_records[record_key] = output["maskmem_pos_enc"]
            if isinstance(output.get("pred_masks"), torch.Tensor):
                preserved_masks[record_key] = output["pred_masks"]

    object_ids = tuple(_object_id(inference_state, index) for index in object_indices)
    metadata = {
        "source": "sam2_inference_state",
        "preserved_pred_masks": preserved_masks,
        "storage_device": str(inference_state.get("storage_device", "unknown")),
        "compute_device": str(inference_state.get("device", "unknown")),
    }
    return CanonicalState(
        spatial_memory=spatial,
        object_pointer=pointer,
        presence_logits=presence,
        frame_indices=frame_indices,
        slot_order=slot_order,
        is_conditioning=is_conditioning,
        validity=validity,
        object_ids=object_ids,
        switch_frame=int(switch_frame),
        positional_information={
            "policy": "source_observed_do_not_translate",
            "records": positional_records,
        },
        metadata=metadata,
    ).validate()


def probe_sam2_inference_state(inference_state: Mapping[str, Any]) -> InspectionReport:
    """Recursively inventory the full predictor container without modifying it."""

    return inspect_state(inference_state, root_name="sam2_inference_state")


def materialize_sam2_history(
    state: CanonicalState,
    *,
    positional_factory: Callable[
        [Any, int, bool, torch.Tensor], list[torch.Tensor] | None
    ],
) -> dict[int, dict[str, dict[int, dict[str, Any]]]]:
    """Materialize the minimal v1.1 history consumed by SAM 2 memory attention.

    ``positional_factory`` must be target-owned.  This intentional boundary keeps
    source positional encodings out of the learned translator and makes an actual
    injection smoke test fail closed when target regeneration is unavailable.

    Historical Source ``pred_masks`` and ``object_score_logits`` are deliberately
    absent. Pinned SAM 2 reads only memory features, their positional encodings and
    object pointers when propagating from ``switch_frame + 1``. A correction on or
    before the switch must use the replay path instead of this translated history.
    """

    state.validate()
    if state.spatial_memory.shape[0] != 1:
        raise ValueError("SAM 2 materialization currently requires B=1")
    result: dict[int, dict[str, dict[int, dict[str, Any]]]] = {}
    for object_slot, object_id in enumerate(state.object_ids):
        result[object_slot] = {COND_KEY: {}, NON_COND_KEY: {}}
        for record_slot in range(state.spatial_memory.shape[2]):
            if not bool(state.validity[0, object_slot, record_slot]):
                continue
            frame = int(state.frame_indices[0, object_slot, record_slot].item())
            is_cond = bool(state.is_conditioning[0, object_slot, record_slot].item())
            feature = state.spatial_memory[0, object_slot, record_slot].unsqueeze(0)
            positional = positional_factory(object_id, frame, is_cond, feature)
            if positional is None:
                raise ValueError(
                    "target positional_factory returned None for a valid memory record"
                )
            output = {
                "maskmem_features": feature,
                "maskmem_pos_enc": positional,
                "obj_ptr": state.object_pointer[0, object_slot, record_slot].unsqueeze(0),
            }
            storage_key = COND_KEY if is_cond else NON_COND_KEY
            result[object_slot][storage_key][frame] = output
    return result


def make_target_sam2_positional_factory(
    predictor: Any,
    inference_state: Mapping[str, Any],
) -> Callable[[Any, int, bool, torch.Tensor], list[torch.Tensor]]:
    """Build target-owned spatial PE without running a past-frame backbone.

    Pinned SAM 2 computes memory positional encoding from the memory grid shape;
    it is constant across frames and objects.  Calling the target memory
    encoder's position module therefore regenerates PE without source PE or a
    video-frame feature.
    """

    position_encoding = predictor.memory_encoder.position_encoding
    compute_device = inference_state["device"]
    constants = inference_state["constants"]

    def factory(
        _object_id: Any,
        _frame: int,
        _is_conditioning: bool,
        feature: torch.Tensor,
    ) -> list[torch.Tensor]:
        cached = constants.get("maskmem_pos_enc")
        expected_grid = tuple(feature.shape[-2:])
        if cached is None:
            device_type = torch.device(compute_device).type
            if torch.is_autocast_enabled(device_type):
                target_dtype = torch.get_autocast_dtype(device_type)
            else:
                parameters = getattr(predictor.memory_encoder, "parameters", None)
                reference_parameter = (
                    next(parameters(), feature) if callable(parameters) else feature
                )
                target_dtype = reference_parameter.dtype
            reference = torch.empty(
                (1, feature.shape[1], *expected_grid),
                device=compute_device,
                dtype=target_dtype,
            )
            generated = position_encoding(reference).to(dtype=target_dtype)
            cached = [generated[0:1].clone()]
            constants["maskmem_pos_enc"] = cached
        actual_grid = tuple(cached[-1].shape[-2:])
        if actual_grid != expected_grid:
            raise ValueError(
                f"target positional grid {actual_grid} does not match memory "
                f"grid {expected_grid}"
            )
        return [value.expand(feature.shape[0], -1, -1, -1) for value in cached]

    return factory


def _move_nested_tensors(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return type(value)(
            (key, _move_nested_tensors(item, device)) for key, item in value.items()
        )
    if isinstance(value, list):
        return [_move_nested_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested_tensors(item, device) for item in value)
    return deepcopy(value)


def inject_sam2_canonical_state(
    state: CanonicalState,
    *,
    predictor: Any,
    inference_state: dict[str, Any],
) -> dict[str, int]:
    """Inject a canonical history into a fresh target predictor state.

    The fresh target state must contain video/runtime-owned fields but no
    registered objects or temporary interactions. Only the v1.1 read-state
    tensors are placed on the target's storage/compute devices, while record
    identity metadata are copied exactly. Prompt/tracking dictionaries start
    empty because they are not required for next-frame continuation. Source
    masks and scores remain in the external CanonicalState archive and are never
    inserted into ``inference_state``. Video length and size are owned by the
    target runtime and are intentionally not compared with Source metadata here.
    """

    state.validate()
    if inference_state.get("obj_ids") or inference_state.get("output_dict_per_obj"):
        raise ValueError("target inference_state must be fresh before injection")
    if any(inference_state.get("temp_output_dict_per_obj", {}).values()):
        raise ValueError("target inference_state contains temporary outputs")
    positional_factory = make_target_sam2_positional_factory(
        predictor, inference_state
    )
    histories = materialize_sam2_history(
        state,
        positional_factory=positional_factory,
    )
    compute_device = torch.device(inference_state["device"])
    storage_device = torch.device(inference_state["storage_device"])
    inference_state["obj_id_to_idx"] = OrderedDict(
        (object_id, object_slot)
        for object_slot, object_id in enumerate(state.object_ids)
    )
    inference_state["obj_idx_to_id"] = OrderedDict(
        (object_slot, object_id)
        for object_slot, object_id in enumerate(state.object_ids)
    )
    inference_state["obj_ids"] = list(state.object_ids)
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}

    for object_slot, _object_id_value in enumerate(state.object_ids):
        history = histories[object_slot]
        for records in history.values():
            for output in records.values():
                output["maskmem_features"] = output["maskmem_features"].to(
                    storage_device
                )
                output["maskmem_pos_enc"] = [
                    value.to(compute_device) for value in output["maskmem_pos_enc"]
                ]
                output["obj_ptr"] = output["obj_ptr"].to(compute_device)
        inference_state["output_dict_per_obj"][object_slot] = history
        inference_state["temp_output_dict_per_obj"][object_slot] = {
            COND_KEY: {},
            NON_COND_KEY: {},
        }
        inference_state["point_inputs_per_obj"][object_slot] = {}
        inference_state["mask_inputs_per_obj"][object_slot] = {}
        inference_state["frames_tracked_per_obj"][object_slot] = {}

    return {
        "objects": len(state.object_ids),
        "records": state.valid_record_count(),
        "switch_frame": state.switch_frame,
    }


def init_sam2_inference_state_without_warmup(
    predictor: Any,
    *,
    video_path: str,
    offload_video_to_cpu: bool = False,
    offload_state_to_cpu: bool = False,
    async_loading_frames: bool = False,
) -> dict[str, Any]:
    """Mirror pinned ``init_state`` while intentionally skipping frame-0 warmup."""

    from sam2.utils.misc import load_video_frames

    compute_device = predictor.device
    images, video_height, video_width = load_video_frames(
        video_path=video_path,
        image_size=predictor.image_size,
        offload_video_to_cpu=offload_video_to_cpu,
        async_loading_frames=async_loading_frames,
        compute_device=compute_device,
    )
    storage_device = torch.device("cpu") if offload_state_to_cpu else compute_device
    return {
        "images": images,
        "num_frames": len(images),
        "offload_video_to_cpu": offload_video_to_cpu,
        "offload_state_to_cpu": offload_state_to_cpu,
        "video_height": video_height,
        "video_width": video_width,
        "device": compute_device,
        "storage_device": storage_device,
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
    }
