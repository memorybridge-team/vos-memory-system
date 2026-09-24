"""Path-independent, checksummed cache for expensive SAM 2 case references."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .state_schema import CanonicalState


SCHEMA_VERSION = "cmmt.prepared_handoff_case.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_masks(masks: Mapping[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {
        int(frame): value.detach().cpu().float()
        for frame, value in sorted(masks.items())
    }


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return type(value)((key, _to_cpu(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return deepcopy(value)


def _cpu_state(state: CanonicalState) -> CanonicalState:
    state.validate()
    return CanonicalState(
        spatial_memory=state.spatial_memory.detach().cpu(),
        object_pointer=state.object_pointer.detach().cpu(),
        presence_logits=state.presence_logits.detach().cpu(),
        frame_indices=state.frame_indices.detach().cpu(),
        slot_order=state.slot_order.detach().cpu(),
        is_conditioning=state.is_conditioning.detach().cpu(),
        validity=state.validity.detach().cpu(),
        object_ids=tuple(state.object_ids),
        switch_frame=state.switch_frame,
        positional_information=_to_cpu(state.positional_information),
        metadata=_to_cpu(state.metadata),
        schema_version=state.schema_version,
    ).validate()


def validate_case_cache(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported case cache: {payload.get('schema_version')!r}")
    source = payload.get("source_canonical")
    target = payload.get("target_canonical")
    if not isinstance(source, CanonicalState) or not isinstance(target, CanonicalState):
        raise TypeError("case cache must contain source and target CanonicalState values")
    source.validate()
    target.validate()
    for label, state in (("source", source), ("target", target)):
        continuous = (state.spatial_memory, state.object_pointer, state.presence_logits)
        if any(tensor.device.type != "cpu" for tensor in continuous):
            raise ValueError(f"{label} canonical state is not CPU-offloaded")
    if source.switch_frame != target.switch_frame:
        raise ValueError("source and target switch frames differ")
    switch_frame = source.switch_frame
    source_masks = payload.get("source_prefix_masks")
    oracle_masks = payload.get("target_oracle_future_masks")
    if not isinstance(source_masks, Mapping) or not source_masks:
        raise ValueError("case cache has no source prefix masks")
    if not isinstance(oracle_masks, Mapping) or not oracle_masks:
        raise ValueError("case cache has no target oracle future masks")
    source_frames = sorted(int(frame) for frame in source_masks)
    oracle_frames = sorted(int(frame) for frame in oracle_masks)
    if source_frames != list(range(0, switch_frame + 1)):
        raise ValueError(
            f"source prefix masks must cover 0..{switch_frame}; got {source_frames}"
        )
    expected_future = list(range(switch_frame + 1, oracle_frames[-1] + 1))
    if oracle_frames != expected_future:
        raise ValueError(f"target oracle future masks are not contiguous: {oracle_frames}")
    for group_name, masks in (
        ("source_prefix_masks", source_masks),
        ("target_oracle_future_masks", oracle_masks),
    ):
        for frame, value in masks.items():
            if not isinstance(frame, int) or not isinstance(value, torch.Tensor):
                raise TypeError(f"{group_name} must map integer frames to tensors")
            if value.device.type != "cpu":
                raise ValueError(f"{group_name}[{frame}] is not CPU-offloaded")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("case cache metadata must be a mapping")
    if "switch_frame" in metadata and int(metadata["switch_frame"]) != switch_frame:
        raise ValueError("case cache metadata switch frame differs from state")
    if "num_frames" in metadata:
        num_frames = int(metadata["num_frames"])
        if num_frames <= switch_frame + 1:
            raise ValueError("case cache metadata leaves no continuation frames")
        if oracle_frames[-1] != num_frames - 1:
            raise ValueError(
                "target oracle future masks do not reach the metadata final frame"
            )
    return payload


def write_case_cache(
    output: str | Path,
    *,
    source_canonical: CanonicalState,
    target_canonical: CanonicalState,
    source_prefix_masks: Mapping[int, torch.Tensor],
    target_oracle_future_masks: Mapping[int, torch.Tensor],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically write a trusted local cache and SHA-256 sidecar."""

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_canonical": _cpu_state(source_canonical),
        "target_canonical": _cpu_state(target_canonical),
        "source_prefix_masks": _cpu_masks(source_prefix_masks),
        "target_oracle_future_masks": _cpu_masks(target_oracle_future_masks),
        "metadata": dict(metadata),
    }
    validate_case_cache(payload)
    partial = output.with_suffix(output.suffix + ".partial")
    torch.save(payload, partial)
    partial.replace(output)
    digest = _sha256(output)
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_partial = checksum_path.with_suffix(checksum_path.suffix + ".partial")
    checksum_partial.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    checksum_partial.replace(checksum_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(output),
        "sha256": digest,
        "bytes": output.stat().st_size,
        "source_prefix_frames": len(source_prefix_masks),
        "target_oracle_future_frames": len(target_oracle_future_masks),
    }


def load_case_cache(path: str | Path, *, verify_checksum: bool = True) -> dict[str, Any]:
    """Load an explicitly trusted cache after optional sidecar verification."""

    path = Path(path).resolve()
    if verify_checksum:
        checksum_path = path.with_suffix(path.suffix + ".sha256")
        if not checksum_path.is_file():
            raise FileNotFoundError(f"case cache checksum is missing: {checksum_path}")
        expected = checksum_path.read_text(encoding="ascii").split()[0]
        actual = _sha256(path)
        if actual != expected:
            raise ValueError("case cache SHA-256 does not match its sidecar")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("case cache payload must be a dictionary")
    validate_case_cache(payload)
    return payload
