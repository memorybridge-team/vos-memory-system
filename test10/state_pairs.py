"""The state-only prepared_handoff_case.v2 contract used by state_pair_examples."""
from dataclasses import replace
from pathlib import Path
import os

from core import read, write, sha
from manifest import library_imports

library_imports()
import torch
from vos_memory_inspector.state_schema import validate_paired_state_contract
from vos_memory_inspector.transformer_translator import SAM21_MEMORY_SPEC

SCHEMA = "cmmt.prepared_handoff_case.v2"
TENSORS = ("spatial_memory", "object_pointer", "presence_logits", "frame_indices",
           "slot_order", "is_conditioning", "validity")
MODELS = {"source_model_id": "sam2.1-small", "target_model_id": "sam2.1-base-plus"}


def active_state(state, *, num_maskmem=7, max_obj_ptrs=16):
    """All conditioning + latest max(num_maskmem-1,max_obj_ptrs-1) non-cond records.

    Match the examples, retaining chronological record order and compact slot_order.
    Batch/object groups with different counts are padded and never enter the loss.
    """
    state.validate()
    limit = max(num_maskmem - 1, max_obj_ptrs - 1)
    if num_maskmem < 1 or max_obj_ptrs < 1:
        raise ValueError("memory limits must be positive")
    b, objects, _ = state.validity.shape
    groups = {}
    for batch in range(b):
        for obj in range(objects):
            valid = state.validity[batch, obj].nonzero().flatten().tolist()
            valid.sort(key=lambda i: int(state.frame_indices[batch, obj, i]))
            cond = [i for i in valid if bool(state.is_conditioning[batch, obj, i])]
            recent = [i for i in valid if not bool(state.is_conditioning[batch, obj, i])]
            selected = set(cond + (recent[-limit:] if limit else []))
            groups[batch, obj] = [i for i in valid if i in selected]
    count = max(map(len, groups.values()))
    if not count:
        raise ValueError("state has no valid records")
    fields = {}
    for name in TENSORS:
        old = getattr(state, name).detach().cpu()
        new = torch.zeros((b, objects, count, *old.shape[3:]), dtype=old.dtype)
        for (batch, obj), indices in groups.items():
            new[batch, obj, :len(indices)] = old[batch, obj, indices]
            if name == "slot_order":
                new[batch, obj, :len(indices)] = torch.arange(len(indices))
        fields[name] = new
    metadata = {k: v for k, v in state.metadata.items() if k != "preserved_pred_masks"}
    metadata.update(storage_device="cpu", sam2_active_memory_selection=dict(
        num_maskmem=num_maskmem, max_obj_ptrs_in_encoder=max_obj_ptrs,
        non_conditioning_limit=limit, policy="all_conditioning_plus_latest_nonconditioning",
        records_before=state.valid_record_count(), records_after=int(fields["validity"].sum())))
    return replace(state, **fields, metadata=metadata).validate()


def validate_pair(payload):
    if payload.get("schema_version") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}; old shard/cache formats are not accepted")
    source, target = payload["source_canonical"], payload["target_canonical"]
    valid = validate_paired_state_contract(source, target)
    if not valid.any():
        raise ValueError("pair has no valid records")
    meta = payload["metadata"]
    if any(meta.get(k) != v for k, v in MODELS.items()):
        raise ValueError("pair must describe SAM 2.1 Small -> Base+")
    if meta.get("cache_mode") != "state_only" or meta.get("active_memory_only") is not True:
        raise ValueError("expected an active-memory state-only pair")
    if meta.get("switch_frame") != source.switch_frame:
        raise ValueError("pair metadata switch_frame mismatch (use processed positions)")
    if int(meta.get("num_frames", 0)) <= source.switch_frame:
        raise ValueError("pair switch is outside the video")
    for state in (source, target):
        if state.spec != SAM21_MEMORY_SPEC:
            raise ValueError("pair tensor dimensions differ from SAM 2.1 memory spec")
        if state.spatial_memory.dtype != torch.bfloat16 or state.object_pointer.dtype != torch.float32:
            raise ValueError("pair requires bf16 spatial memory and fp32 pointers")
        if state.presence_logits.dtype != torch.float32:
            raise ValueError("pair requires fp32 diagnostic presence logits")
        if state.frame_indices.dtype != torch.int64 or state.slot_order.dtype != torch.int64:
            raise ValueError("pair requires int64 frame_indices and slot_order")
        for name in TENSORS:
            tensor = getattr(state, name)
            if tensor.device.type != "cpu":
                raise ValueError("pair tensors must be CPU offloaded")
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"nonfinite {name}")
        indices = state.frame_indices[state.validity]
        if (indices < 0).any() or (indices > state.switch_frame).any():
            raise ValueError("pair contains records outside the prefix")
        limit = max(int(meta["num_maskmem"]) - 1, int(meta["max_obj_ptrs_in_encoder"]) - 1)
        if int(meta["num_maskmem"]) < 1 or int(meta["max_obj_ptrs_in_encoder"]) < 1:
            raise ValueError("pair memory limits must be positive")
        for batch in range(state.validity.shape[0]):
            for obj in range(state.validity.shape[1]):
                v = state.validity[batch, obj]
                frames = state.frame_indices[batch, obj][v].tolist()
                if frames != sorted(set(frames)):
                    raise ValueError("pair record frames must be unique and chronological")
                if int((v & ~state.is_conditioning[batch, obj]).sum()) > limit:
                    raise ValueError("pair exceeds the active non-conditioning record limit")
    return source, target, meta


def load_pair(path, *, expected_sha=None):
    path = Path(path)
    checksum = path.with_suffix(".pt.sha256")
    expected = checksum.read_text(encoding="ascii").split()[0]
    actual = sha(path)
    if actual != expected or (expected_sha and actual != expected_sha):
        raise ValueError(f"pair checksum mismatch: {path}")
    # Local/prepared research caches contain CanonicalState objects; do not load untrusted files.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result = validate_pair(payload)
    prepare = read(path.with_suffix(".prepare.json"))
    if prepare["cache"]["schema_version"] != SCHEMA or prepare["cache"]["sha256"] != actual:
        raise ValueError(f"pair prepare sidecar mismatch: {path}")
    for field in (*MODELS, "video_id", "object_id", "switch_frame", "num_frames", "upstream_commit"):
        if prepare.get(field) != result[2].get(field):
            raise ValueError(f"prepare metadata mismatch: {field}")
    return result


def save_pair(path, source, target, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema_version=SCHEMA, source_canonical=source, target_canonical=target,
                   metadata=dict(metadata))
    validate_pair(payload)
    temp = path.with_suffix(".pt.partial")
    torch.save(payload, temp)
    os.replace(temp, path)
    checksum = sha(path)
    sidecar = path.with_suffix(".pt.sha256")
    temp = sidecar.with_suffix(".partial")
    temp.write_text(f"{checksum}  {path.name}\n", encoding="ascii")
    os.replace(temp, sidecar)
    write(path.with_suffix(".prepare.json"), dict(metadata,
        source_contract=source.contract_dict(), target_contract=target.contract_dict(),
        cache=dict(schema_version=SCHEMA, sha256=checksum, bytes=path.stat().st_size,
                   cache_mode="state_only", source_prefix_frames=0, target_oracle_future_frames=0)))
    return checksum


def verify_case_pair(case, source, metadata):
    if metadata["video_id"] != case["video_id"] or int(metadata["object_id"]) != case["object_id"]:
        raise ValueError("pair and evaluation video/object differ")
    if source.switch_frame != case["switch"] or metadata["num_frames"] != len(case["frame_stems"]):
        raise ValueError("pair and evaluation frame timeline differ")
