"""Runtime-validated canonical handoff state for CMMT experiments.

The schema deliberately separates continuous tensors that a translator may learn
from discrete bookkeeping that must be copied exactly.  Shapes are discovered at
runtime; the axis contracts below are the only fixed part of the representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from typing import Any, Mapping

import torch


SPATIAL_AXES = ("batch", "object", "record", "channel", "height", "width")
POINTER_AXES = ("batch", "object", "record", "feature")
SCALAR_AXES = ("batch", "object", "record", "scalar")
RECORD_AXES = ("batch", "object", "record")


@dataclass(frozen=True)
class StateSpec:
    """Continuous target contract used by shape adapters and translators."""

    feature_channels: int
    height: int
    width: int
    pointer_dim: int
    positional_policy: str = "regenerate_at_target"

    def __post_init__(self) -> None:
        values = {
            "feature_channels": self.feature_channels,
            "height": self.height,
            "width": self.width,
            "pointer_dim": self.pointer_dim,
        }
        invalid = {name: value for name, value in values.items() if value <= 0}
        if invalid:
            raise ValueError(f"StateSpec dimensions must be positive: {invalid}")
        if self.positional_policy not in {"regenerate_at_target", "provided_target"}:
            raise ValueError(
                "positional_policy must be 'regenerate_at_target' or 'provided_target'"
            )

    @classmethod
    def from_state(cls, state: "CanonicalState") -> "StateSpec":
        state.validate()
        _, _, _, channels, height, width = state.spatial_memory.shape
        pointer_dim = state.object_pointer.shape[-1]
        policy = str(
            state.positional_information.get("policy", "regenerate_at_target")
        )
        if policy not in {"regenerate_at_target", "provided_target"}:
            policy = "regenerate_at_target"
        return cls(channels, height, width, pointer_dim, policy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_channels": self.feature_channels,
            "height": self.height,
            "width": self.width,
            "pointer_dim": self.pointer_dim,
            "positional_policy": self.positional_policy,
        }


@dataclass
class CanonicalState:
    """Continuation-oriented state at switch time ``t``.

    ``record`` is a stable, padded index over conditioning and non-conditioning
    frame records.  ``validity`` distinguishes real records from padding.
    Continuous data are translated; record identity and ordering are not.
    """

    spatial_memory: torch.Tensor
    object_pointer: torch.Tensor
    presence_logits: torch.Tensor
    frame_indices: torch.Tensor
    slot_order: torch.Tensor
    is_conditioning: torch.Tensor
    validity: torch.Tensor
    object_ids: tuple[Any, ...]
    switch_frame: int
    positional_information: dict[str, Any] = field(
        default_factory=lambda: {"policy": "regenerate_at_target"}
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "cmmt.canonical_state.v1"

    def validate(self) -> "CanonicalState":
        if not isinstance(self.spatial_memory, torch.Tensor):
            raise TypeError("spatial_memory must be a torch.Tensor")
        if self.spatial_memory.ndim != 6:
            raise ValueError(
                "spatial_memory must have axes [B,O,K,C,H,W], got "
                f"{tuple(self.spatial_memory.shape)}"
            )
        b, objects, records, _, _, _ = self.spatial_memory.shape
        expected_record_shape = (b, objects, records)

        expected = {
            "object_pointer": (4, expected_record_shape),
            "presence_logits": (4, expected_record_shape),
            "frame_indices": (3, expected_record_shape),
            "slot_order": (3, expected_record_shape),
            "is_conditioning": (3, expected_record_shape),
            "validity": (3, expected_record_shape),
        }
        for name, (ndim, prefix) in expected.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.ndim != ndim or tuple(value.shape[:3]) != prefix:
                raise ValueError(
                    f"{name} must start with [B,O,K]={prefix}; got "
                    f"{tuple(value.shape)}"
                )
        if self.presence_logits.shape[-1] != 1:
            raise ValueError("presence_logits must have a scalar final axis")
        if self.is_conditioning.dtype != torch.bool:
            raise TypeError("is_conditioning must use torch.bool")
        if self.validity.dtype != torch.bool:
            raise TypeError("validity must use torch.bool")
        if len(self.object_ids) != objects:
            raise ValueError(
                f"object_ids has {len(self.object_ids)} entries for O={objects}"
            )
        if not isinstance(self.switch_frame, int):
            raise TypeError("switch_frame must be an int")
        if records < 1:
            raise ValueError("canonical state needs at least one record slot")
        return self

    @property
    def spec(self) -> StateSpec:
        return StateSpec.from_state(self)

    def continuous_bytes(self) -> int:
        """Bytes in the archived continuous tensors, including diagnostics.

        This is deliberately not the research handoff-byte metric because
        ``presence_logits`` is retained for analysis but is not injected.
        """

        self.validate()
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.spatial_memory,
                self.object_pointer,
                self.presence_logits,
            )
        )

    def handoff_bytes(self) -> int:
        """Bytes in the minimal continuation payload used by the target.

        The payload contains the two translated tensors plus the discrete
        assembly fields. Target-generated positional encodings, diagnostic
        tensors, prompts, video properties, and dataset fingerprints are not
        counted. Object IDs are counted using their deterministic JSON UTF-8
        representation and ``switch_frame`` as a signed 64-bit integer.
        """

        self.validate()
        tensor_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.spatial_memory,
                self.object_pointer,
                self.frame_indices,
                self.slot_order,
                self.is_conditioning,
                self.validity,
            )
        )
        object_id_bytes = len(
            json.dumps(
                list(self.object_ids),
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        return tensor_bytes + object_id_bytes + 8

    def valid_record_count(self) -> int:
        return int(self.validity.sum().item())

    def with_continuous(
        self,
        *,
        spatial_memory: torch.Tensor,
        object_pointer: torch.Tensor,
        presence_logits: torch.Tensor,
        positional_information: Mapping[str, Any] | None = None,
        translation_metadata: Mapping[str, Any] | None = None,
    ) -> "CanonicalState":
        """Return translated tensors while copying all discrete state exactly."""

        metadata = dict(self.metadata)
        if translation_metadata:
            metadata["translation"] = dict(translation_metadata)
        result = replace(
            self,
            spatial_memory=spatial_memory,
            object_pointer=object_pointer,
            presence_logits=presence_logits,
            frame_indices=self.frame_indices.clone(),
            slot_order=self.slot_order.clone(),
            is_conditioning=self.is_conditioning.clone(),
            validity=self.validity.clone(),
            object_ids=tuple(self.object_ids),
            positional_information=dict(
                positional_information
                if positional_information is not None
                else {"policy": "regenerate_at_target"}
            ),
            metadata=metadata,
        )
        return result.validate()

    def contract_dict(self) -> dict[str, Any]:
        """JSON-safe schema/shape summary; tensor values are intentionally omitted."""

        self.validate()
        tensors = {
            "spatial_memory": (self.spatial_memory, SPATIAL_AXES, "translate"),
            "object_pointer": (self.object_pointer, POINTER_AXES, "translate"),
            "presence_logits": (self.presence_logits, SCALAR_AXES, "calibrate"),
            "frame_indices": (self.frame_indices, RECORD_AXES, "preserve"),
            "slot_order": (self.slot_order, RECORD_AXES, "preserve"),
            "is_conditioning": (self.is_conditioning, RECORD_AXES, "preserve"),
            "validity": (self.validity, RECORD_AXES, "preserve"),
        }
        return {
            "schema_version": self.schema_version,
            "switch_frame": self.switch_frame,
            "object_ids": [str(value) for value in self.object_ids],
            "valid_record_count": self.valid_record_count(),
            "continuous_bytes": self.continuous_bytes(),
            "handoff_bytes": self.handoff_bytes(),
            "tensors": {
                name: {
                    "shape": list(tensor.shape),
                    "axes": list(axes),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "bytes": tensor.numel() * tensor.element_size(),
                    "policy": policy,
                }
                for name, (tensor, axes, policy) in tensors.items()
            },
            "positional_information": {
                "policy": self.positional_information.get(
                    "policy", "regenerate_at_target"
                ),
                "keys": sorted(self.positional_information),
            },
            "metadata_keys": sorted(self.metadata),
        }


def validate_paired_state_contract(
    source: CanonicalState,
    target: CanonicalState,
) -> torch.Tensor:
    """Fail closed unless a paired source/target state describes one timeline.

    Continuous feature dimensions may differ across models.  Object identity,
    record identity, conditioning roles and padding may not: otherwise the
    trainer would compare unrelated memories while still producing a loss.
    The returned validity mask is safe to use for component losses.
    """

    source.validate()
    target.validate()
    if source.schema_version != target.schema_version:
        raise ValueError("paired source/target schema_version must match exactly")
    if source.spatial_memory.shape[:3] != target.spatial_memory.shape[:3]:
        raise ValueError("paired source/target [B,O,K] axes must match")
    if source.object_ids != target.object_ids:
        raise ValueError("paired source/target object_ids must match exactly")
    if source.switch_frame != target.switch_frame:
        raise ValueError("paired source/target switch_frame must match exactly")

    discrete_tensors = (
        "frame_indices",
        "slot_order",
        "is_conditioning",
        "validity",
    )
    for name in discrete_tensors:
        source_value = getattr(source, name).detach().cpu()
        target_value = getattr(target, name).detach().cpu()
        if not torch.equal(source_value, target_value):
            raise ValueError(f"paired source/target {name} must match exactly")

    return source.validity
