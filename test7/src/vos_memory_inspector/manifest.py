from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


MANIFEST_FIELDS = (
    "model_id",
    "upstream_commit",
    "video_id",
    "frame_idx",
    "switch_frame",
    "object_id",
    "cond_non_cond",
    "tensor_name",
    "producer",
    "consumer",
    "state_path",
    "temporal_class",
    "shape",
    "dtype",
    "device",
    "mean",
    "std",
    "norm",
    "min",
    "max",
    "num_bytes",
    "changed_from_previous",
    "max_abs_delta_from_previous",
    "dump_path",
)


def flatten_tensors(name: str, value: Any) -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield name, value
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from flatten_tensors(f"{name}[{index}]", item)


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    numeric = detached.float()
    if numeric.numel() == 0:
        mean = std = norm = minimum = maximum = None
    else:
        mean = float(numeric.mean().item())
        std = float(numeric.std(unbiased=False).item())
        norm = float(torch.linalg.vector_norm(numeric).item())
        minimum = float(numeric.min().item())
        maximum = float(numeric.max().item())
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "mean": mean,
        "std": std,
        "norm": norm,
        "min": minimum,
        "max": maximum,
        "num_bytes": detached.numel() * detached.element_size(),
    }


class TensorChangeTracker:
    """Exact consecutive comparison while retaining one detached tensor per key."""

    def __init__(self) -> None:
        self._previous: dict[tuple[Any, ...], torch.Tensor] = {}

    def compare(
        self, key: tuple[Any, ...], tensor: torch.Tensor
    ) -> tuple[bool | None, float | None]:
        current = tensor.detach()
        previous = self._previous.get(key)
        changed: bool | None = None
        max_abs_delta: float | None = None
        if previous is not None:
            compatible = previous.shape == current.shape and previous.dtype == current.dtype
            if compatible:
                if previous.device != current.device:
                    previous = previous.to(current.device)
                changed = not torch.equal(previous, current)
                if current.numel() and (current.is_floating_point() or current.is_complex()):
                    max_abs_delta = float(
                        (current - previous).abs().max().float().item()
                    )
                elif current.numel():
                    max_abs_delta = float((current != previous).any().item())
                else:
                    max_abs_delta = 0.0
            else:
                changed = True
        self._previous[key] = current
        return changed, max_abs_delta


@dataclass(frozen=True)
class DumpPolicy:
    directory: Path | None = None
    selected_names: frozenset[str] = frozenset()

    def wants(self, tensor_name: str) -> bool:
        return self.directory is not None and any(
            tensor_name == name or tensor_name.startswith(f"{name}[")
            for name in self.selected_names
        )

    def dump(
        self,
        *,
        tensor: torch.Tensor,
        model_id: str,
        video_id: str,
        object_id: int | str,
        frame_idx: int,
        tensor_name: str,
    ) -> str | None:
        if not self.wants(tensor_name):
            return None
        assert self.directory is not None
        safe_model = _safe_component(model_id)
        safe_video = _safe_component(video_id)
        safe_object = _safe_component(str(object_id))
        safe_tensor = _safe_component(tensor_name)
        path = (
            self.directory
            / safe_model
            / safe_video
            / f"object-{safe_object}"
            / f"frame-{frame_idx:06d}-{safe_tensor}.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor.detach().cpu().clone(), path)
        return str(path.resolve())


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "unnamed"


class ManifestWriter:
    def __init__(self, jsonl_path: str | Path, csv_path: str | Path | None = None):
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._jsonl = self.jsonl_path.open("w", encoding="utf-8", newline="\n")
        self._csv_file = None
        self._csv_writer = None
        if csv_path is not None:
            csv_path = Path(csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = csv_path.open("w", encoding="utf-8", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=MANIFEST_FIELDS)
            self._csv_writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        normalized = {field: _json_safe(row.get(field)) for field in MANIFEST_FIELDS}
        self._jsonl.write(json.dumps(normalized, sort_keys=False) + "\n")
        self._jsonl.flush()
        if self._csv_writer is not None:
            csv_row = dict(normalized)
            csv_row["shape"] = json.dumps(csv_row["shape"])
            self._csv_writer.writerow(csv_row)
            assert self._csv_file is not None
            self._csv_file.flush()

    def close(self) -> None:
        self._jsonl.close()
        if self._csv_file is not None:
            self._csv_file.close()

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

