from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .inventory import FRAME_STATE_KEYS, FRAME_TENSOR_ROUTES
from .manifest import (
    DumpPolicy,
    ManifestWriter,
    TensorChangeTracker,
    flatten_tensors,
    tensor_stats,
)


@dataclass(frozen=True)
class ProbeConfig:
    model_id: str
    upstream_commit: str
    video_id: str
    switch_frame: int


class StateProbe:
    """Inspect compact per-object outputs already committed to inference_state."""

    def __init__(
        self,
        config: ProbeConfig,
        writer: ManifestWriter,
        *,
        dump_policy: DumpPolicy | None = None,
        change_tracker: TensorChangeTracker | None = None,
    ) -> None:
        self.config = config
        self.writer = writer
        self.dump_policy = dump_policy or DumpPolicy()
        self.change_tracker = change_tracker or TensorChangeTracker()

    def record_frame(
        self,
        inference_state: Mapping[str, Any],
        frame_idx: int,
        *,
        object_ids: list[int | str] | None = None,
        strict: bool = True,
    ) -> int:
        obj_id_to_idx = inference_state["obj_id_to_idx"]
        ids = list(obj_id_to_idx) if object_ids is None else object_ids
        rows_written = 0
        for object_id in ids:
            if object_id not in obj_id_to_idx:
                raise KeyError(f"Unknown object_id={object_id!r}")
            object_idx = obj_id_to_idx[object_id]
            outputs = inference_state["output_dict_per_obj"][object_idx]
            frame_output = outputs["cond_frame_outputs"].get(frame_idx)
            cond_non_cond = "cond"
            storage_key = "cond_frame_outputs"
            if frame_output is None:
                frame_output = outputs["non_cond_frame_outputs"].get(frame_idx)
                cond_non_cond = "non_cond"
                storage_key = "non_cond_frame_outputs"
            if frame_output is None:
                if strict:
                    raise KeyError(
                        f"No compact output for object_id={object_id!r}, frame={frame_idx}"
                    )
                continue

            missing = [key for key in FRAME_STATE_KEYS if key not in frame_output]
            if missing:
                raise KeyError(
                    f"Frame output for object_id={object_id!r}, frame={frame_idx} "
                    f"is missing {missing}"
                )
            for base_name in FRAME_STATE_KEYS:
                route = FRAME_TENSOR_ROUTES[base_name]
                value = frame_output[base_name]
                for tensor_name, tensor in flatten_tensors(base_name, value):
                    changed, max_abs_delta = self.change_tracker.compare(
                        (object_id, tensor_name), tensor
                    )
                    dump_path = self.dump_policy.dump(
                        tensor=tensor,
                        model_id=self.config.model_id,
                        video_id=self.config.video_id,
                        object_id=object_id,
                        frame_idx=frame_idx,
                        tensor_name=tensor_name,
                    )
                    row = {
                        "model_id": self.config.model_id,
                        "upstream_commit": self.config.upstream_commit,
                        "video_id": self.config.video_id,
                        "frame_idx": int(frame_idx),
                        "switch_frame": int(self.config.switch_frame),
                        "object_id": object_id,
                        "cond_non_cond": cond_non_cond,
                        "tensor_name": tensor_name,
                        "producer": route.producer,
                        "consumer": route.consumer,
                        "state_path": (
                            f'inference_state["output_dict_per_obj"][{object_idx}]'
                            f'["{storage_key}"][{frame_idx}]["{base_name}"]'
                        ),
                        "temporal_class": route.temporal_class,
                        **tensor_stats(tensor),
                        "changed_from_previous": changed,
                        "max_abs_delta_from_previous": max_abs_delta,
                        "dump_path": dump_path,
                    }
                    self.writer.write(row)
                    rows_written += 1
        return rows_written

