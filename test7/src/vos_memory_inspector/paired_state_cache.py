"""Compact checksummed cache containing only an aligned canonical-state pair."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .state_schema import CanonicalState


SCHEMA_VERSION = "cmmt.paired_state_cache.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_paired_state_cache(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported paired-state cache: {payload.get('schema_version')!r}")
    source = payload.get("source_canonical")
    target = payload.get("target_canonical")
    if not isinstance(source, CanonicalState) or not isinstance(target, CanonicalState):
        raise TypeError("paired-state cache needs source and target CanonicalState values")
    source.validate()
    target.validate()
    if source.switch_frame != target.switch_frame:
        raise ValueError("paired source/target switch frames differ")
    if source.object_ids != target.object_ids:
        raise ValueError("paired source/target object IDs differ")
    if source.spatial_memory.shape[:3] != target.spatial_memory.shape[:3]:
        raise ValueError("paired source/target [B,O,K] axes differ")
    for state_name, state in (("source", source), ("target", target)):
        for tensor in (
            state.spatial_memory,
            state.object_pointer,
            state.presence_logits,
        ):
            if tensor.device.type != "cpu":
                raise ValueError(f"{state_name} state is not CPU-offloaded")
    if not isinstance(payload.get("metadata"), Mapping):
        raise TypeError("paired-state cache metadata must be a mapping")
    return payload


def write_paired_state_cache(
    output: str | Path,
    *,
    source_canonical: CanonicalState,
    target_canonical: CanonicalState,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_canonical": source_canonical,
        "target_canonical": target_canonical,
        "metadata": dict(metadata),
    }
    validate_paired_state_cache(payload)
    partial = output.with_suffix(output.suffix + ".partial")
    torch.save(payload, partial)
    partial.replace(output)
    digest = _sha256(output)
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum_partial = checksum.with_suffix(checksum.suffix + ".partial")
    checksum_partial.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    checksum_partial.replace(checksum)
    return {"path": str(output), "sha256": digest, "bytes": output.stat().st_size}


def load_paired_state_cache(
    path: str | Path, *, verify_checksum: bool = True
) -> tuple[CanonicalState, CanonicalState, Mapping[str, Any]]:
    path = Path(path).resolve()
    if verify_checksum:
        checksum = path.with_suffix(path.suffix + ".sha256")
        if not checksum.is_file():
            raise FileNotFoundError(f"paired-state checksum is missing: {checksum}")
        expected = checksum.read_text(encoding="ascii").split()[0]
        if _sha256(path) != expected:
            raise ValueError("paired-state cache SHA-256 does not match its sidecar")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("paired-state cache payload must be a mapping")
    validate_paired_state_cache(payload)
    return (
        payload["source_canonical"],
        payload["target_canonical"],
        payload["metadata"],
    )
