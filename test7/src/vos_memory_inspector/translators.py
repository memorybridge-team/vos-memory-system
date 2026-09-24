"""Component-wise CMMT baselines for SAM 2-style canonical state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .state_schema import CanonicalState, StateSpec, validate_paired_state_contract


def _resample_spatial(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tensor.ndim != 6:
        raise ValueError("spatial input must be [B,O,K,C,H,W]")
    if tensor.shape[-2:] == (height, width):
        return tensor
    batch, objects, records, channels, old_h, old_w = tensor.shape
    flat = tensor.reshape(batch * objects * records, channels, old_h, old_w)
    resized = F.interpolate(flat, size=(height, width), mode="bilinear", align_corners=False)
    return resized.reshape(batch, objects, records, channels, height, width)


def _pad_or_truncate(tensor: torch.Tensor, output_dim: int, axis: int) -> torch.Tensor:
    axis = axis if axis >= 0 else tensor.ndim + axis
    current = tensor.shape[axis]
    if current == output_dim:
        return tensor.clone()
    if current > output_dim:
        slices = [slice(None)] * tensor.ndim
        slices[axis] = slice(0, output_dim)
        return tensor[tuple(slices)].clone()
    shape = list(tensor.shape)
    shape[axis] = output_dim - current
    padding = tensor.new_zeros(shape)
    return torch.cat((tensor, padding), dim=axis)


def _target_positional(spec: StateSpec) -> dict[str, str]:
    return {"policy": spec.positional_policy}


def _pair_guard(source: CanonicalState, target: CanonicalState) -> torch.Tensor:
    return validate_paired_state_contract(source, target)


class DirectCopyTranslator:
    """No-learning baseline with explicit grid resize and zero-pad/truncate rules."""

    name = "direct"

    def __init__(self, target_spec: StateSpec):
        self.target_spec = target_spec

    def translate(self, source: CanonicalState) -> CanonicalState:
        source.validate()
        spatial = _resample_spatial(
            source.spatial_memory, self.target_spec.height, self.target_spec.width
        )
        spatial = _pad_or_truncate(
            spatial, self.target_spec.feature_channels, axis=3
        )
        pointer = _pad_or_truncate(
            source.object_pointer, self.target_spec.pointer_dim, axis=-1
        )
        return source.with_continuous(
            spatial_memory=spatial,
            object_pointer=pointer,
            presence_logits=source.presence_logits.clone(),
            positional_information=_target_positional(self.target_spec),
            translation_metadata={
                "translator": self.name,
                "channel_adapter": "truncate_or_zero_pad",
                "grid_adapter": "bilinear",
            },
        )

    def parameter_count(self) -> int:
        return 0


@dataclass(frozen=True)
class AffineMap:
    weight: torch.Tensor
    bias: torch.Tensor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=tensor.device, dtype=tensor.dtype)
        bias = self.bias.to(device=tensor.device, dtype=tensor.dtype)
        return tensor @ weight + bias

    @property
    def parameter_count(self) -> int:
        return self.weight.numel() + self.bias.numel()


def fit_affine_closed_form(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge_lambda: float,
    fit_dtype: torch.dtype = torch.float64,
) -> AffineMap:
    """Fit centered affine OLS (lambda=0) or ridge in closed form."""

    if inputs.ndim != 2 or targets.ndim != 2 or inputs.shape[0] != targets.shape[0]:
        raise ValueError("affine calibration expects X=[N,Din], Y=[N,Dout]")
    if inputs.shape[0] < 1:
        raise ValueError("affine calibration has no observations")
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must be non-negative")
    x = inputs.detach().to(device="cpu", dtype=fit_dtype)
    y = targets.detach().to(device="cpu", dtype=fit_dtype)
    mean_x = x.mean(dim=0, keepdim=True)
    mean_y = y.mean(dim=0, keepdim=True)
    xc = x - mean_x
    yc = y - mean_y
    if ridge_lambda == 0:
        weight = torch.linalg.lstsq(xc, yc).solution
    else:
        gram = xc.T @ xc
        identity = torch.eye(gram.shape[0], dtype=gram.dtype)
        weight = torch.linalg.solve(
            gram + float(ridge_lambda) * identity, xc.T @ yc
        )
    bias = (mean_y - mean_x @ weight).squeeze(0)
    output_dtype = inputs.dtype if inputs.is_floating_point() else torch.float32
    return AffineMap(weight.to(output_dtype), bias.to(output_dtype))


class RidgeStateTranslator:
    name = "ridge"

    def __init__(
        self,
        target_spec: StateSpec,
        feature_map: AffineMap,
        pointer_map: AffineMap,
        ridge_lambda: float,
    ):
        self.target_spec = target_spec
        self.feature_map = feature_map
        self.pointer_map = pointer_map
        self.ridge_lambda = ridge_lambda

    @classmethod
    def fit(
        cls,
        pairs: Iterable[tuple[CanonicalState, CanonicalState]],
        *,
        ridge_lambda: float = 0.01,
    ) -> "RidgeStateTranslator":
        pair_list = list(pairs)
        if not pair_list:
            raise ValueError("at least one paired state is required")
        target_spec = pair_list[0][1].spec
        feature_x: list[torch.Tensor] = []
        feature_y: list[torch.Tensor] = []
        pointer_x: list[torch.Tensor] = []
        pointer_y: list[torch.Tensor] = []
        for source, target in pair_list:
            if target.spec != target_spec:
                raise ValueError("all target states must share one runtime StateSpec")
            valid = _pair_guard(source, target)
            src_spatial = _resample_spatial(
                source.spatial_memory, target_spec.height, target_spec.width
            )
            sx = src_spatial.permute(0, 1, 2, 4, 5, 3)[valid]
            sy = target.spatial_memory.permute(0, 1, 2, 4, 5, 3)[
                valid.to(target.validity.device)
            ]
            feature_x.append(sx.reshape(-1, sx.shape[-1]))
            feature_y.append(sy.reshape(-1, sy.shape[-1]))
            pointer_x.append(source.object_pointer[valid])
            pointer_y.append(target.object_pointer[valid.to(target.validity.device)])
        return cls(
            target_spec=target_spec,
            feature_map=fit_affine_closed_form(
                torch.cat(feature_x), torch.cat(feature_y), ridge_lambda=ridge_lambda
            ),
            pointer_map=fit_affine_closed_form(
                torch.cat(pointer_x), torch.cat(pointer_y), ridge_lambda=ridge_lambda
            ),
            ridge_lambda=ridge_lambda,
        )

    def translate(self, source: CanonicalState) -> CanonicalState:
        source.validate()
        spatial = _resample_spatial(
            source.spatial_memory, self.target_spec.height, self.target_spec.width
        )
        spatial = self.feature_map(spatial.movedim(3, -1)).movedim(-1, 3)
        pointer = self.pointer_map(source.object_pointer)
        return source.with_continuous(
            spatial_memory=spatial,
            object_pointer=pointer,
            presence_logits=source.presence_logits.clone(),
            positional_information=_target_positional(self.target_spec),
            translation_metadata={
                "translator": self.name,
                "ridge_lambda": self.ridge_lambda,
                "grid_adapter": "bilinear",
            },
        )

    def parameter_count(self) -> int:
        return self.feature_map.parameter_count + self.pointer_map.parameter_count

    def to_payload(self) -> dict[str, Any]:
        return {
            "target_spec": self.target_spec.to_dict(),
            "ridge_lambda": self.ridge_lambda,
            "feature_weight": self.feature_map.weight.detach().cpu(),
            "feature_bias": self.feature_map.bias.detach().cpu(),
            "pointer_weight": self.pointer_map.weight.detach().cpu(),
            "pointer_bias": self.pointer_map.bias.detach().cpu(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RidgeStateTranslator":
        required = {
            "target_spec",
            "ridge_lambda",
            "feature_weight",
            "feature_bias",
            "pointer_weight",
            "pointer_bias",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"Ridge payload is missing fields: {missing}")
        spec_value = payload["target_spec"]
        if not isinstance(spec_value, Mapping):
            raise TypeError("Ridge target_spec must be a mapping")
        target_spec = StateSpec(**dict(spec_value))
        tensor_fields = {
            name: payload[name]
            for name in required
            if name.endswith("weight") or name.endswith("bias")
        }
        invalid = [name for name, value in tensor_fields.items() if not isinstance(value, torch.Tensor)]
        if invalid:
            raise TypeError(f"Ridge payload fields must be tensors: {invalid}")
        return cls(
            target_spec=target_spec,
            feature_map=AffineMap(payload["feature_weight"], payload["feature_bias"]),
            pointer_map=AffineMap(payload["pointer_weight"], payload["pointer_bias"]),
            ridge_lambda=float(payload["ridge_lambda"]),
        )


class RidgeDirectPresenceTranslator:
    """Use Ridge for memory/pointer and preserve the source presence logit."""

    name = "ridge_spatial_pointer_direct_presence"

    def __init__(self, ridge: RidgeStateTranslator):
        self.ridge = ridge
        self.target_spec = ridge.target_spec

    def translate(self, source: CanonicalState) -> CanonicalState:
        ridge_state = self.ridge.translate(source)
        return ridge_state.with_continuous(
            spatial_memory=ridge_state.spatial_memory,
            object_pointer=ridge_state.object_pointer,
            presence_logits=source.presence_logits.clone(),
            positional_information=_target_positional(self.target_spec),
            translation_metadata={
                "translator": self.name,
                "ridge_lambda": self.ridge.ridge_lambda,
                "spatial_memory": "ridge",
                "object_pointer": "ridge",
                "presence_logits": "direct",
                "grid_adapter": "bilinear",
            },
        )

    def parameter_count(self) -> int:
        return self.ridge.feature_map.parameter_count + self.ridge.pointer_map.parameter_count


class _LearnedStateTranslator(nn.Module):
    name = "learned"

    def __init__(self, source_spec: StateSpec, target_spec: StateSpec):
        super().__init__()
        self.source_spec = source_spec
        self.target_spec = target_spec

    def _feature_head(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _pointer_head(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, source: CanonicalState) -> CanonicalState:
        source.validate()
        if source.spec != self.source_spec:
            raise ValueError(
                f"source runtime spec {source.spec} does not match {self.source_spec}"
            )
        spatial = _resample_spatial(
            source.spatial_memory, self.target_spec.height, self.target_spec.width
        )
        spatial = self._feature_head(spatial.movedim(3, -1)).movedim(-1, 3)
        pointer = self._pointer_head(source.object_pointer)
        return source.with_continuous(
            spatial_memory=spatial,
            object_pointer=pointer,
            presence_logits=source.presence_logits.clone(),
            positional_information=_target_positional(self.target_spec),
            translation_metadata={
                "translator": self.name,
                "grid_adapter": "bilinear",
            },
        )

    translate = forward

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class LinearStateTranslator(_LearnedStateTranslator):
    name = "linear"

    def __init__(self, source_spec: StateSpec, target_spec: StateSpec):
        super().__init__(source_spec, target_spec)
        self.feature = nn.Linear(
            source_spec.feature_channels, target_spec.feature_channels
        )
        self.pointer = nn.Linear(source_spec.pointer_dim, target_spec.pointer_dim)

    def _feature_head(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.feature(
            tensor.to(device=self.feature.weight.device, dtype=self.feature.weight.dtype)
        )

    def _pointer_head(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.pointer(
            tensor.to(device=self.pointer.weight.device, dtype=self.pointer.weight.dtype)
        )

class ResidualMLPStateTranslator(_LearnedStateTranslator):
    """Separate two-layer MLP heads; identity residuals only when shapes match."""

    name = "residual_mlp"

    def __init__(
        self,
        source_spec: StateSpec,
        target_spec: StateSpec,
        *,
        hidden_dim: int = 128,
    ):
        super().__init__(source_spec, target_spec)
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.feature = nn.Sequential(
            nn.Linear(source_spec.feature_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_spec.feature_channels),
        )
        self.pointer = nn.Sequential(
            nn.Linear(source_spec.pointer_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_spec.pointer_dim),
        )
        self.feature_residual = (
            source_spec.feature_channels == target_spec.feature_channels
            and source_spec.height == target_spec.height
            and source_spec.width == target_spec.width
        )
        self.pointer_residual = source_spec.pointer_dim == target_spec.pointer_dim

    def _feature_head(self, tensor: torch.Tensor) -> torch.Tensor:
        calibrated = tensor.to(
            device=self.feature[0].weight.device,
            dtype=self.feature[0].weight.dtype,
        )
        output = self.feature(calibrated)
        return output + calibrated if self.feature_residual else output

    def _pointer_head(self, tensor: torch.Tensor) -> torch.Tensor:
        calibrated = tensor.to(
            device=self.pointer[0].weight.device,
            dtype=self.pointer[0].weight.dtype,
        )
        output = self.pointer(calibrated)
        return output + calibrated if self.pointer_residual else output

    def to_payload(self) -> dict[str, Any]:
        """Serialize architecture metadata with CPU weights for safe reuse."""

        return {
            "schema_version": "cmmt.residual_mlp_translator.v2",
            "translator": self.name,
            "source_spec": self.source_spec.to_dict(),
            "target_spec": self.target_spec.to_dict(),
            "hidden_dim": self.hidden_dim,
            "state_dict": {
                name: value.detach().cpu() for name, value in self.state_dict().items()
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ResidualMLPStateTranslator":
        if payload.get("schema_version") != "cmmt.residual_mlp_translator.v2":
            raise ValueError("unsupported residual MLP translator payload")
        source_spec = StateSpec(**dict(payload["source_spec"]))
        target_spec = StateSpec(**dict(payload["target_spec"]))
        translator = cls(
            source_spec,
            target_spec,
            hidden_dim=int(payload["hidden_dim"]),
        )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise TypeError("residual MLP payload state_dict must be a mapping")
        translator.load_state_dict(dict(state_dict), strict=True)
        translator.eval()
        return translator


class LearnedComponentPolicyTranslator:
    """Use selected learned continuous components and Direct Copy for the rest."""

    _ALLOWED_COMPONENTS = {
        "spatial_memory",
        "object_pointer",
    }

    def __init__(
        self,
        learned: _LearnedStateTranslator,
        *,
        learned_components: tuple[str, ...],
    ):
        selected = frozenset(learned_components)
        if not selected or not selected <= self._ALLOWED_COMPONENTS:
            raise ValueError(
                "learned_components must be a non-empty subset of "
                f"{sorted(self._ALLOWED_COMPONENTS)}"
            )
        self.learned = learned
        self.learned_components = selected
        self.target_spec = learned.target_spec
        self.name = "learned_" + "_".join(
            component
            for component in sorted(self._ALLOWED_COMPONENTS)
            if component in selected
        )

    def translate(self, source: CanonicalState) -> CanonicalState:
        learned_state = self.learned.translate(source)
        direct_state = DirectCopyTranslator(self.target_spec).translate(source)

        def choose(name: str) -> torch.Tensor:
            learned_tensor = getattr(learned_state, name)
            if name in self.learned_components:
                return learned_tensor
            return getattr(direct_state, name).to(
                device=learned_tensor.device,
                dtype=learned_tensor.dtype,
            )

        return learned_state.with_continuous(
            spatial_memory=choose("spatial_memory"),
            object_pointer=choose("object_pointer"),
            presence_logits=choose("presence_logits"),
            positional_information=_target_positional(self.target_spec),
            translation_metadata={
                "translator": self.name,
                "component_policy": {
                    component: (
                        "learned" if component in self.learned_components else "direct"
                    )
                    for component in sorted(self._ALLOWED_COMPONENTS)
                },
                "presence_logits": "diagnostic_only",
                "grid_adapter": "bilinear",
            },
        )

    def parameter_count(self) -> int:
        return self.learned.parameter_count()

