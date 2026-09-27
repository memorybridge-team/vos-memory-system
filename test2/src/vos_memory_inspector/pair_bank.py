"""Checksummed record bank for aligned and native Small/Base+ state pairs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .state_schema import CanonicalState, validate_paired_state_contract


SCHEMA_VERSION = "cmmt.state_pair_bank.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    partial.replace(path)


@dataclass(frozen=True)
class SemanticRecordKey:
    dataset: str
    release: str
    split: str
    pair_type: str
    video_id: str
    object_id: str
    frame_idx: int
    is_conditioning: bool

    def __post_init__(self) -> None:
        if self.pair_type not in {"aligned", "native"}:
            raise ValueError("pair_type must be aligned or native")
        if self.split not in {"fit", "dev", "test"}:
            raise ValueError("split must be fit, dev or test")

    @property
    def record_id(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "release": self.release,
            "split": self.split,
            "pair_type": self.pair_type,
            "video_id": self.video_id,
            "object_id": self.object_id,
            "frame_idx": self.frame_idx,
            "is_conditioning": self.is_conditioning,
        }


def slice_record(state: CanonicalState, object_slot: int, record_slot: int) -> CanonicalState:
    """Copy one immutable record while preserving the canonical metadata contract."""

    state.validate()
    if not bool(state.validity[0, object_slot, record_slot]):
        raise ValueError("cannot slice a padded canonical record")
    result = CanonicalState(
        spatial_memory=state.spatial_memory[:, object_slot : object_slot + 1, record_slot : record_slot + 1].detach().cpu().clone(),
        object_pointer=state.object_pointer[:, object_slot : object_slot + 1, record_slot : record_slot + 1].detach().cpu().clone(),
        presence_logits=state.presence_logits[:, object_slot : object_slot + 1, record_slot : record_slot + 1].detach().cpu().clone(),
        frame_indices=state.frame_indices[:, object_slot : object_slot + 1, record_slot : record_slot + 1].detach().cpu().clone(),
        slot_order=torch.zeros((1, 1, 1), dtype=state.slot_order.dtype),
        is_conditioning=state.is_conditioning[:, object_slot : object_slot + 1, record_slot : record_slot + 1].detach().cpu().clone(),
        validity=torch.ones((1, 1, 1), dtype=torch.bool),
        object_ids=(state.object_ids[object_slot],),
        switch_frame=state.switch_frame,
        positional_information=dict(state.positional_information),
        metadata={key: value for key, value in state.metadata.items() if key != "preserved_pred_masks"},
    )
    return result.validate()


class PairBankWriter:
    def __init__(
        self,
        root: str | Path,
        *,
        dataset: str,
        release: str,
        pair_type: str,
        split: str,
        provenance: Mapping[str, Any],
    ) -> None:
        self.root = Path(root).resolve() / dataset / pair_type / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.release = release
        self.pair_type = pair_type
        self.split = split
        self.provenance = dict(provenance)
        self.index_path = self.root / "index.json"
        if self.index_path.is_file():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._validate_index()
        else:
            self.index = {
                "schema_version": SCHEMA_VERSION,
                "dataset": dataset,
                "release": release,
                "pair_type": pair_type,
                "split": split,
                "provenance": self.provenance,
                "records": {},
                "snapshots": {},
                "failed_pairs": [],
            }

    def _validate_index(self) -> None:
        expected = {
            "schema_version": SCHEMA_VERSION,
            "dataset": self.dataset,
            "release": self.release,
            "pair_type": self.pair_type,
            "split": self.split,
        }
        for name, value in expected.items():
            if self.index.get(name) != value:
                raise ValueError(f"pair bank {name} mismatch: {self.index.get(name)!r} != {value!r}")
        if self.index.get("provenance") != self.provenance:
            raise ValueError("pair bank provenance/checkpoint contract changed")

    def write_record(
        self,
        key: SemanticRecordKey,
        source: CanonicalState,
        target: CanonicalState,
        *,
        frame_id: str | int,
        predictor_frame_idx: int,
        training_eligible: bool = True,
    ) -> str:
        validate_paired_state_contract(source, target)
        if source.valid_record_count() != 1 or target.valid_record_count() != 1:
            raise ValueError("bank records must contain exactly one canonical record")
        actual_frame = int(source.frame_indices[0, 0, 0])
        actual_cond = bool(source.is_conditioning[0, 0, 0])
        if actual_frame != predictor_frame_idx or actual_cond != key.is_conditioning:
            raise ValueError("semantic key does not match canonical frame/conditioning role")
        record_id = key.record_id
        relative = Path("records") / record_id[:2] / f"{record_id}.pt"
        output = self.root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "cmmt.state_pair_record.v1",
            "semantic_key": key.to_dict(),
            "frame_id": str(frame_id),
            "predictor_frame_idx": predictor_frame_idx,
            "source": source,
            "target": target,
        }
        if output.is_file():
            digest = _sha256(output)
        else:
            partial = output.with_suffix(".pt.partial")
            torch.save(payload, partial)
            partial.replace(output)
            digest = _sha256(output)
            output.with_suffix(".pt.sha256").write_text(
                f"{digest}  {output.name}\n", encoding="ascii"
            )
        existing = self.index["records"].get(record_id)
        row = {
            "path": relative.as_posix(),
            "sha256": digest,
            "semantic_key": key.to_dict(),
            "frame_id": str(frame_id),
            "predictor_frame_idx": predictor_frame_idx,
            "training_eligible": bool(training_eligible),
        }
        if existing is not None:
            existing_normalized = dict(existing)
            existing_training = bool(existing_normalized.pop("training_eligible", False))
            row_without_training = dict(row)
            row_without_training.pop("training_eligible")
            if existing_normalized != row_without_training:
                raise ValueError(f"record {record_id} already exists with different metadata")
            row["training_eligible"] = existing_training or bool(training_eligible)
        self.index["records"][record_id] = row
        self.flush()
        return record_id

    def write_snapshot(
        self,
        *,
        video_id: str,
        object_id: str,
        switch_frame: int,
        record_ids: list[str],
        frame_id_map: Mapping[str, int],
    ) -> str:
        missing = sorted(set(record_ids) - set(self.index["records"]))
        if missing:
            raise ValueError(f"snapshot references missing records: {missing[:3]}")
        snapshot_id = hashlib.sha256(
            f"{video_id}:{object_id}:{switch_frame}:{self.pair_type}".encode("utf-8")
        ).hexdigest()
        row = {
            "video_id": video_id,
            "object_id": object_id,
            "switch_frame": switch_frame,
            "record_ids": list(record_ids),
            "frame_id_map": {str(key): int(value) for key, value in frame_id_map.items()},
        }
        existing = self.index["snapshots"].get(snapshot_id)
        if existing is not None and existing != row:
            raise ValueError("snapshot ID collision with different content")
        self.index["snapshots"][snapshot_id] = row
        self.flush()
        return snapshot_id

    def log_failure(self, payload: Mapping[str, Any]) -> None:
        self.index["failed_pairs"].append(dict(payload))
        self.flush()

    def flush(self) -> None:
        _atomic_json(self.index_path, self.index)


def load_bank_record(path: str | Path, *, expected_sha256: str | None = None) -> tuple[CanonicalState, CanonicalState, Mapping[str, Any]]:
    path = Path(path).resolve()
    digest = _sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("pair bank record SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "cmmt.state_pair_record.v1":
        raise ValueError("unsupported pair bank record")
    source = payload["source"].validate()
    target = payload["target"].validate()
    validate_paired_state_contract(source, target)
    return source, target, payload


def load_pair_bank(index_path: str | Path) -> list[tuple[CanonicalState, CanonicalState]]:
    index_path = Path(index_path).resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported pair bank index")
    pairs = []
    for record_id in sorted(index["records"]):
        row = index["records"][record_id]
        if not bool(row.get("training_eligible", True)):
            continue
        source, target, _payload = load_bank_record(
            index_path.parent / row["path"], expected_sha256=row["sha256"]
        )
        identity = {
            "dataset": index["dataset"],
            "release": index["release"],
            "pair_type": index["pair_type"],
            "split": index["split"],
            "video_id": row["semantic_key"]["video_id"],
            "object_id": row["semantic_key"]["object_id"],
            "frame_idx": row["semantic_key"]["frame_idx"],
        }
        source.metadata.update(identity)
        target.metadata.update(identity)
        pairs.append((source, target))
    return pairs


def _assemble_snapshot_state(
    records: list[CanonicalState],
    *,
    switch_frame: int,
    snapshot_id: str,
    frame_id_map: Mapping[str, int],
) -> CanonicalState:
    if not records:
        raise ValueError("snapshot must reference at least one record")
    first = records[0].validate()
    if any(record.object_ids != first.object_ids for record in records):
        raise ValueError("snapshot records must describe the same object")
    if any(record.spatial_memory.shape[:2] != (1, 1) for record in records):
        raise ValueError("snapshot records must be single-batch, single-object states")
    frame_indices = torch.cat([record.frame_indices for record in records], dim=2)
    if bool((frame_indices > switch_frame).any()):
        raise ValueError("snapshot references a memory record after its switch frame")
    count = len(records)
    state = CanonicalState(
        spatial_memory=torch.cat([record.spatial_memory for record in records], dim=2),
        object_pointer=torch.cat([record.object_pointer for record in records], dim=2),
        presence_logits=torch.cat([record.presence_logits for record in records], dim=2),
        frame_indices=frame_indices,
        slot_order=torch.arange(count, dtype=first.slot_order.dtype).view(1, 1, count),
        is_conditioning=torch.cat([record.is_conditioning for record in records], dim=2),
        validity=torch.ones((1, 1, count), dtype=torch.bool),
        object_ids=first.object_ids,
        switch_frame=int(switch_frame),
        positional_information=dict(first.positional_information),
        metadata={
            **dict(first.metadata),
            "snapshot_id": snapshot_id,
            "frame_id_map": {str(key): int(value) for key, value in frame_id_map.items()},
            "immutable_record_count": count,
        },
    )
    return state.validate()


def load_bank_snapshot(
    index_path: str | Path, snapshot_id: str
) -> tuple[CanonicalState, CanonicalState, Mapping[str, Any]]:
    """Rebuild a continuation state from checksummed immutable record references."""

    index_path = Path(index_path).resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported pair bank index")
    try:
        snapshot = index["snapshots"][snapshot_id]
    except KeyError as exc:
        raise KeyError(f"unknown pair-bank snapshot {snapshot_id}") from exc
    source_records: list[CanonicalState] = []
    target_records: list[CanonicalState] = []
    for record_id in snapshot["record_ids"]:
        try:
            row = index["records"][record_id]
        except KeyError as exc:
            raise ValueError(f"snapshot references unknown record {record_id}") from exc
        source, target, _ = load_bank_record(
            index_path.parent / row["path"], expected_sha256=row["sha256"]
        )
        source_records.append(source)
        target_records.append(target)
    kwargs = {
        "switch_frame": int(snapshot["switch_frame"]),
        "snapshot_id": snapshot_id,
        "frame_id_map": snapshot["frame_id_map"],
    }
    source_state = _assemble_snapshot_state(source_records, **kwargs)
    target_state = _assemble_snapshot_state(target_records, **kwargs)
    validate_paired_state_contract(source_state, target_state)
    return source_state, target_state, snapshot
