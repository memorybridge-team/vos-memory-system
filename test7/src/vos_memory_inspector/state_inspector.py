"""Framework-light recursive tensor-state inspection and reporting."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class TensorRecord:
    path: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    bytes: int
    requires_grad: bool
    axes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "shape": list(self.shape),
            "axes": list(self.axes),
            "dtype": self.dtype,
            "device": self.device,
            "numel": self.numel,
            "bytes": self.bytes,
            "requires_grad": self.requires_grad,
        }


@dataclass(frozen=True)
class ContainerRecord:
    path: str
    kind: str
    length: int | None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind, "length": self.length}


@dataclass(frozen=True)
class InspectionReport:
    root_type: str
    tensors: tuple[TensorRecord, ...]
    containers: tuple[ContainerRecord, ...]
    truncated_paths: tuple[str, ...] = ()

    @property
    def total_bytes(self) -> int:
        return sum(record.bytes for record in self.tensors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cmmt.state_inspection.v1",
            "root_type": self.root_type,
            "tensor_count": len(self.tensors),
            "total_bytes": self.total_bytes,
            "tensors": [record.to_dict() for record in self.tensors],
            "containers": [record.to_dict() for record in self.containers],
            "truncated_paths": list(self.truncated_paths),
        }


def _known_axes(path: str, ndim: int) -> tuple[str, ...]:
    lowered = path.lower()
    if lowered.endswith(".spatial_memory") and ndim == 6:
        return ("batch", "object", "record", "channel", "height", "width")
    if lowered.endswith(".object_pointer") and ndim == 4:
        return ("batch", "object", "record", "feature")
    if lowered.endswith(".presence_logits") and ndim == 4:
        return ("batch", "object", "record", "scalar")
    if any(
        lowered.endswith(f".{name}")
        for name in ("frame_indices", "slot_order", "is_conditioning", "validity")
    ) and ndim == 3:
        return ("batch", "object", "record")
    if any(part in lowered for part in ("key_cache", "value_cache", ".key", ".value")):
        if ndim == 4:
            return ("batch", "kv_head", "sequence", "head_dim")
    if "maskmem_features" in lowered and ndim == 4:
        return ("object_or_batch", "channel", "height", "width")
    if "maskmem_pos_enc" in lowered and ndim == 4:
        return ("object_or_batch", "channel", "height", "width")
    if "obj_ptr" in lowered and ndim == 2:
        return ("object_or_batch", "feature")
    if "object_score_logits" in lowered and ndim == 2:
        return ("object_or_batch", "scalar")
    if "pred_masks" in lowered and ndim == 4:
        return ("object_or_batch", "mask_channel", "height", "width")
    return ()


def inspect_state(
    value: Any,
    *,
    root_name: str = "state",
    max_depth: int = 32,
) -> InspectionReport:
    """Inspect tensors inside nested mappings, sequences, dataclasses and caches.

    Cache objects are handled by public, duck-typed attributes only.  Arbitrary
    objects are not recursively introspected through every private attribute,
    preventing accidental traversal of a complete model graph.
    """

    tensors: list[TensorRecord] = []
    containers: list[ContainerRecord] = []
    truncated: list[str] = []
    active_ids: set[int] = set()

    def visit(current: Any, path: str, depth: int) -> None:
        if isinstance(current, torch.Tensor):
            tensors.append(
                TensorRecord(
                    path=path,
                    shape=tuple(current.shape),
                    dtype=str(current.dtype),
                    device=str(current.device),
                    numel=current.numel(),
                    bytes=current.numel() * current.element_size(),
                    requires_grad=current.requires_grad,
                    axes=_known_axes(path, current.ndim),
                )
            )
            return
        if current is None or isinstance(current, (str, bytes, int, float, bool)):
            return
        if depth >= max_depth:
            truncated.append(path)
            return

        identity = id(current)
        if identity in active_ids:
            truncated.append(f"{path} (cycle)")
            return
        active_ids.add(identity)
        try:
            if isinstance(current, Mapping):
                containers.append(ContainerRecord(path, type(current).__name__, len(current)))
                for key, child in current.items():
                    visit(child, f"{path}.{key}", depth + 1)
                return
            if dataclasses.is_dataclass(current) and not isinstance(current, type):
                fields = dataclasses.fields(current)
                containers.append(ContainerRecord(path, type(current).__name__, len(fields)))
                for item in fields:
                    visit(getattr(current, item.name), f"{path}.{item.name}", depth + 1)
                return
            if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
                containers.append(ContainerRecord(path, type(current).__name__, len(current)))
                for index, child in enumerate(current):
                    visit(child, f"{path}[{index}]", depth + 1)
                return

            # Hugging Face cache APIs changed over time.  These public layouts
            # cover the legacy lists, current cache lists and CacheLayer objects.
            if hasattr(current, "key_cache") and hasattr(current, "value_cache"):
                containers.append(ContainerRecord(path, type(current).__name__, None))
                visit(getattr(current, "key_cache"), f"{path}.key_cache", depth + 1)
                visit(getattr(current, "value_cache"), f"{path}.value_cache", depth + 1)
                return
            if hasattr(current, "layers"):
                containers.append(ContainerRecord(path, type(current).__name__, None))
                visit(getattr(current, "layers"), f"{path}.layers", depth + 1)
                return
            if hasattr(current, "keys") and hasattr(current, "values"):
                containers.append(ContainerRecord(path, type(current).__name__, 2))
                visit(getattr(current, "keys"), f"{path}.keys", depth + 1)
                visit(getattr(current, "values"), f"{path}.values", depth + 1)
                return
            truncated.append(f"{path} ({type(current).__name__})")
        finally:
            active_ids.remove(identity)

    visit(value, root_name, 0)
    return InspectionReport(
        root_type=type(value).__name__,
        tensors=tuple(tensors),
        containers=tuple(containers),
        truncated_paths=tuple(truncated),
    )


def write_inspection_report(
    report: InspectionReport,
    json_path: Path,
    markdown_path: Path | None = None,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if markdown_path is None:
        return
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tensor state inspection",
        "",
        f"- Root type: `{report.root_type}`",
        f"- Tensor count: `{len(report.tensors)}`",
        f"- Tensor bytes: `{report.total_bytes}`",
        "",
        "| Path | Shape | Axes | Dtype | Device | Bytes |",
        "|---|---:|---|---|---|---:|",
    ]
    for record in report.tensors:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | {} |".format(
                record.path,
                list(record.shape),
                list(record.axes),
                record.dtype,
                record.device,
                record.bytes,
            )
        )
    if report.truncated_paths:
        lines.extend(["", "## Not traversed", ""])
        lines.extend(f"- `{path}`" for path in report.truncated_paths)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
