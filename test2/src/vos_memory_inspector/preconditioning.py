"""Fixed canonical transforms and component-wise state translators.

The learned mapping only sees canonical coordinates.  Every transform stores
both directions explicitly so training can always optimize reconstruction in
the target model's raw state space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .state_schema import CanonicalState, StateSpec, validate_paired_state_contract


Pair = tuple[CanonicalState, CanonicalState]


@dataclass(frozen=True)
class FixedCanonicalTransform:
    """Row-vector affine canonicalization for one state component."""

    name: str
    source_offset: torch.Tensor
    source_forward: torch.Tensor
    target_offset: torch.Tensor
    target_forward: torch.Tensor
    target_inverse: torch.Tensor
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tensors = (
            self.source_offset,
            self.source_forward,
            self.target_offset,
            self.target_forward,
            self.target_inverse,
        )
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("canonical transform values must be tensors")
        source_dim = self.source_offset.numel()
        target_dim = self.target_offset.numel()
        if self.source_forward.shape[0] != source_dim:
            raise ValueError("source_forward input dimension does not match offset")
        if self.target_forward.shape[0] != target_dim:
            raise ValueError("target_forward input dimension does not match offset")
        if self.target_inverse.shape != (
            self.target_forward.shape[1],
            target_dim,
        ):
            raise ValueError("target_inverse has incompatible canonical dimension")

    @property
    def source_dim(self) -> int:
        return int(self.source_offset.numel())

    @property
    def canonical_source_dim(self) -> int:
        return int(self.source_forward.shape[1])

    @property
    def canonical_target_dim(self) -> int:
        return int(self.target_forward.shape[1])

    @property
    def target_dim(self) -> int:
        return int(self.target_offset.numel())

    def source_to_canonical(self, value: torch.Tensor) -> torch.Tensor:
        offset = self.source_offset.to(value.device, torch.float32)
        matrix = self.source_forward.to(value.device, torch.float32)
        return (value.float() - offset) @ matrix

    def target_to_canonical(self, value: torch.Tensor) -> torch.Tensor:
        offset = self.target_offset.to(value.device, torch.float32)
        matrix = self.target_forward.to(value.device, torch.float32)
        return (value.float() - offset) @ matrix

    def canonical_to_target(self, value: torch.Tensor) -> torch.Tensor:
        matrix = self.target_inverse.to(value.device, torch.float32)
        offset = self.target_offset.to(value.device, torch.float32)
        return value.float() @ matrix + offset

    def roundtrip_metrics(
        self, value: torch.Tensor, *, deployment_dtype: torch.dtype | None = None
    ) -> dict[str, Any]:
        original = value.float()
        reconstructed = self.canonical_to_target(self.target_to_canonical(original))
        difference = reconstructed - original
        result: dict[str, Any] = {
            "max_abs": float(difference.abs().max().cpu()),
            "mean_abs": float(difference.abs().mean().cpu()),
            "relative_l2": float(
                difference.norm().div(original.norm().clamp_min(1e-12)).cpu()
            ),
            "cosine": float(
                F.cosine_similarity(
                    reconstructed.reshape(1, -1), original.reshape(1, -1), eps=1e-12
                ).cpu()
            ),
            "finite": bool(torch.isfinite(reconstructed).all()),
        }
        if deployment_dtype is not None:
            deployed = self.canonical_to_target(
                self.target_to_canonical(value.to(deployment_dtype))
            ).to(deployment_dtype)
            deployed_error = deployed.float() - original
            result["deployment_dtype"] = str(deployment_dtype)
            result["deployment_max_abs"] = float(deployed_error.abs().max().cpu())
            result["deployment_mean_abs"] = float(deployed_error.abs().mean().cpu())
            result["deployment_finite"] = bool(torch.isfinite(deployed).all())
        return result

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "cmmt.fixed_canonical_transform.v1",
            "name": self.name,
            "source_offset": self.source_offset.detach().cpu(),
            "source_forward": self.source_forward.detach().cpu(),
            "target_offset": self.target_offset.detach().cpu(),
            "target_forward": self.target_forward.detach().cpu(),
            "target_inverse": self.target_inverse.detach().cpu(),
            "settings": dict(self.settings),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FixedCanonicalTransform":
        if payload.get("schema_version") != "cmmt.fixed_canonical_transform.v1":
            raise ValueError("unsupported canonical transform payload")
        return cls(
            name=str(payload["name"]),
            source_offset=payload["source_offset"],
            source_forward=payload["source_forward"],
            target_offset=payload["target_offset"],
            target_forward=payload["target_forward"],
            target_inverse=payload["target_inverse"],
            settings=dict(payload.get("settings", {})),
        )


class _RunningCovariance:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.count = 0
        self.sum = torch.zeros(dimension, dtype=torch.float64)
        self.cross = torch.zeros(dimension, dimension, dtype=torch.float64)

    def update(self, rows: torch.Tensor) -> None:
        rows = rows.detach().to(device="cpu", dtype=torch.float64).reshape(-1, self.dimension)
        if rows.numel() == 0:
            return
        self.count += rows.shape[0]
        self.sum += rows.sum(0)
        self.cross += rows.T @ rows

    def finish(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count < 2:
            raise ValueError("at least two observations are required for statistics")
        mean = self.sum / self.count
        covariance = (self.cross - self.count * torch.outer(mean, mean)) / (
            self.count - 1
        )
        return mean.float(), covariance.float()


def _iter_component_rows(
    pairs: Iterable[Pair], component: str
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    if component not in {"spatial_memory", "object_pointer"}:
        raise ValueError(f"unsupported component: {component}")
    for source, target in pairs:
        valid = validate_paired_state_contract(source, target)
        if component == "object_pointer":
            yield source.object_pointer[valid], target.object_pointer[valid]
            continue
        for batch, obj, record in valid.nonzero(as_tuple=False).tolist():
            source_rows = source.spatial_memory[batch, obj, record].movedim(0, -1)
            target_rows = target.spatial_memory[batch, obj, record].movedim(0, -1)
            yield source_rows.reshape(-1, source_rows.shape[-1]), target_rows.reshape(
                -1, target_rows.shape[-1]
            )


def identity_transform(source_dim: int, target_dim: int) -> FixedCanonicalTransform:
    if source_dim != target_dim:
        raise ValueError("raw canonicalization requires equal source/target dimensions")
    identity = torch.eye(source_dim)
    return FixedCanonicalTransform(
        "raw",
        torch.zeros(source_dim),
        identity,
        torch.zeros(target_dim),
        identity.clone(),
        identity.clone(),
    )


def fit_standardization(
    pairs: Sequence[Pair], component: str, *, epsilon: float = 1e-5
) -> FixedCanonicalTransform:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    first = pairs[0]
    source_dim = (
        first[0].spec.feature_channels
        if component == "spatial_memory"
        else first[0].spec.pointer_dim
    )
    target_dim = (
        first[1].spec.feature_channels
        if component == "spatial_memory"
        else first[1].spec.pointer_dim
    )
    source_stats = _RunningCovariance(source_dim)
    target_stats = _RunningCovariance(target_dim)
    for source_rows, target_rows in _iter_component_rows(pairs, component):
        source_stats.update(source_rows)
        target_stats.update(target_rows)
    source_mean, source_cov = source_stats.finish()
    target_mean, target_cov = target_stats.finish()
    source_scale = source_cov.diag().clamp_min(0).sqrt() + epsilon
    target_scale = target_cov.diag().clamp_min(0).sqrt() + epsilon
    return FixedCanonicalTransform(
        "standardization",
        source_mean,
        torch.diag(source_scale.reciprocal()),
        target_mean,
        torch.diag(target_scale.reciprocal()),
        torch.diag(target_scale),
        {"epsilon": epsilon, "fit_split_only": True},
    )


def fit_zca(
    pairs: Sequence[Pair],
    component: str,
    *,
    shrinkage: float = 1e-3,
    eigenvalue_floor: float = 1e-4,
) -> FixedCanonicalTransform:
    if not 0 <= shrinkage < 1:
        raise ValueError("shrinkage must be in [0, 1)")
    if eigenvalue_floor <= 0:
        raise ValueError("eigenvalue_floor must be positive")
    first = pairs[0]
    source_dim = first[0].spec.feature_channels if component == "spatial_memory" else first[0].spec.pointer_dim
    target_dim = first[1].spec.feature_channels if component == "spatial_memory" else first[1].spec.pointer_dim
    source_stats = _RunningCovariance(source_dim)
    target_stats = _RunningCovariance(target_dim)
    for source_rows, target_rows in _iter_component_rows(pairs, component):
        source_stats.update(source_rows)
        target_stats.update(target_rows)
    source_mean, source_cov = source_stats.finish()
    target_mean, target_cov = target_stats.finish()

    def matrices(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        average_variance = covariance.diag().mean()
        covariance = (1 - shrinkage) * covariance + shrinkage * average_variance * torch.eye(covariance.shape[0])
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
        floor = max(float(eigenvalues.max()) * eigenvalue_floor, torch.finfo(torch.float64).eps)
        stable = eigenvalues.clamp_min(floor)
        forward = eigenvectors @ torch.diag(stable.rsqrt()) @ eigenvectors.T
        inverse = eigenvectors @ torch.diag(stable.sqrt()) @ eigenvectors.T
        return forward.float(), inverse.float(), {
            "raw_min_eigenvalue": float(eigenvalues.min()),
            "floor": floor,
            "condition_number_after_floor": float(stable.max() / stable.min()),
        }

    source_forward, _, source_diagnostic = matrices(source_cov)
    target_forward, target_inverse, target_diagnostic = matrices(target_cov)
    return FixedCanonicalTransform(
        "zca",
        source_mean,
        source_forward,
        target_mean,
        target_forward,
        target_inverse,
        {
            "shrinkage": shrinkage,
            "eigenvalue_floor": eigenvalue_floor,
            "source": source_diagnostic,
            "target": target_diagnostic,
        },
    )


def consumer_metric_transform(
    source_metric: torch.Tensor,
    target_metric: torch.Tensor,
    *,
    source_mean: torch.Tensor | None = None,
    target_mean: torch.Tensor | None = None,
    regularization: float = 1e-5,
    alpha: float = 1.0,
) -> FixedCanonicalTransform:
    if regularization <= 0:
        raise ValueError("regularization must be positive")

    def factor(metric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        metric = metric.detach().double()
        metric = (metric + metric.T) / 2 + regularization * torch.eye(metric.shape[0], dtype=metric.dtype)
        values, vectors = torch.linalg.eigh(metric)
        values = values.clamp_min(regularization)
        forward = vectors @ torch.diag(values.sqrt())
        inverse = torch.diag(values.rsqrt()) @ vectors.T
        return forward.float(), inverse.float(), float(values.max() / values.min())

    source_forward, _, source_condition = factor(source_metric)
    target_forward, target_inverse, target_condition = factor(target_metric)
    source_dim = source_metric.shape[0]
    target_dim = target_metric.shape[0]
    return FixedCanonicalTransform(
        "consumer_projection_aware",
        torch.zeros(source_dim) if source_mean is None else source_mean.float(),
        source_forward,
        torch.zeros(target_dim) if target_mean is None else target_mean.float(),
        target_forward,
        target_inverse,
        {
            "regularization": regularization,
            "alpha": alpha,
            "source_condition_number": source_condition,
            "target_condition_number": target_condition,
        },
    )


def output_projection_svd_transform(
    source_weight: torch.Tensor,
    source_bias: torch.Tensor,
    target_weight: torch.Tensor,
    target_bias: torch.Tensor,
    *,
    relative_floor: float = 1e-4,
) -> FixedCanonicalTransform:
    """Canonicalize only coordinates preserved by the final output projection."""

    if relative_floor <= 0:
        raise ValueError("relative_floor must be positive")

    def factors(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        matrix = weight.detach().float().reshape(weight.shape[0], -1)
        u, singular, _vh = torch.linalg.svd(matrix, full_matrices=False)
        floor = float(singular.max()) * relative_floor
        stable = singular.clamp_min(floor)
        forward = u @ torch.diag(stable.reciprocal())
        inverse = torch.diag(stable) @ u.T
        # With more outputs than inputs, U has fewer columns than rows and
        # U @ U.T is a projection: coordinates outside the column space are lost.
        truncated = matrix.shape[0] > matrix.shape[1]
        regularized = bool(torch.any(singular < floor))
        return forward, inverse, {
            "raw_rank": int(torch.linalg.matrix_rank(matrix)),
            "floor": floor,
            "raw_min_singular": float(singular.min()),
            "raw_max_singular": float(singular.max()),
            "floor_regularized": regularized,
            "column_space_truncated": truncated,
            "lossy_or_regularized": regularized or truncated,
        }

    source_forward, _, source_info = factors(source_weight)
    target_forward, target_inverse, target_info = factors(target_weight)
    return FixedCanonicalTransform(
        "output_projection_svd",
        source_bias.detach().float().reshape(-1),
        source_forward,
        target_bias.detach().float().reshape(-1),
        target_forward,
        target_inverse,
        {
            "relative_floor": relative_floor,
            "source": source_info,
            "target": target_info,
            "lossy_or_regularized": bool(
                source_info["lossy_or_regularized"] or target_info["lossy_or_regularized"]
            ),
            "interpretation": "final_output_projection_coordinates_only",
        },
    )


def anchor_transform(source_anchor: torch.Tensor, target_anchor: torch.Tensor) -> FixedCanonicalTransform:
    source_anchor = source_anchor.detach().float().reshape(-1)
    target_anchor = target_anchor.detach().float().reshape(-1)
    return FixedCanonicalTransform(
        "no_object_anchor",
        source_anchor,
        torch.eye(source_anchor.numel()),
        target_anchor,
        torch.eye(target_anchor.numel()),
        torch.eye(target_anchor.numel()),
        {
            "anchor_preserving_head_required": True,
            "source_anchor": source_anchor,
            "target_anchor": target_anchor,
        },
    )


class _ResidualHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim)
        )
        self.residual = input_dim == output_dim

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.network(value)
        return output + value if self.residual else output


class _AnchorPreservingHead(nn.Module):
    """Guarantee H(0)=0 for fixed no-object anchors."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros((1, value.shape[-1]), device=value.device, dtype=value.dtype)
        return self.linear(value) + self.network(value) - self.network(zero)


class PreconditionedStateTranslator(nn.Module):
    """Separate spatial/pointer heads wrapped by fixed model-specific transforms."""

    name = "preconditioned_component_translator"

    def __init__(
        self,
        source_spec: StateSpec,
        target_spec: StateSpec,
        spatial_transform: FixedCanonicalTransform,
        pointer_transform: FixedCanonicalTransform,
        *,
        architecture: str = "mlp",
        spatial_hidden_dim: int = 128,
        pointer_hidden_dim: int = 512,
        pointer_anchor_preserving: bool = False,
        fit_manifest_sha256: str | None = None,
    ) -> None:
        super().__init__()
        if architecture not in {"linear", "mlp"}:
            raise ValueError("architecture must be linear or mlp")
        self.source_spec = source_spec
        self.target_spec = target_spec
        self.spatial_transform = spatial_transform
        self.pointer_transform = pointer_transform
        self.architecture = architecture
        self.spatial_hidden_dim = spatial_hidden_dim
        self.pointer_hidden_dim = pointer_hidden_dim
        self.pointer_anchor_preserving = pointer_anchor_preserving
        self.fit_manifest_sha256 = fit_manifest_sha256

        if architecture == "linear":
            self.spatial_head = nn.Linear(
                spatial_transform.canonical_source_dim,
                spatial_transform.canonical_target_dim,
            )
            self.pointer_head = nn.Linear(
                pointer_transform.canonical_source_dim,
                pointer_transform.canonical_target_dim,
                bias=not pointer_anchor_preserving,
            )
        else:
            self.spatial_head = _ResidualHead(
                spatial_transform.canonical_source_dim,
                spatial_hidden_dim,
                spatial_transform.canonical_target_dim,
            )
            pointer_cls = _AnchorPreservingHead if pointer_anchor_preserving else _ResidualHead
            self.pointer_head = pointer_cls(
                pointer_transform.canonical_source_dim,
                pointer_hidden_dim,
                pointer_transform.canonical_target_dim,
            )

    def transform_spatial(self, rows: torch.Tensor) -> torch.Tensor:
        canonical = self.spatial_transform.source_to_canonical(rows)
        parameter = next(self.spatial_head.parameters())
        mapped = self.spatial_head(canonical.to(parameter.device, parameter.dtype))
        return self.spatial_transform.canonical_to_target(mapped)

    def transform_pointer(self, rows: torch.Tensor) -> torch.Tensor:
        source_anchor = self.pointer_transform.settings.get("source_anchor")
        target_anchor = self.pointer_transform.settings.get("target_anchor")
        bypass = bool(self.pointer_transform.settings.get("bypass_exact_anchor", False))
        anchor_mask = None
        if bypass:
            if not isinstance(source_anchor, torch.Tensor) or not isinstance(target_anchor, torch.Tensor):
                raise ValueError("exact anchor bypass requires source and target anchors")
            # Exact matching is intentional: with ``fixed_no_obj_ptr`` SAM 2.1
            # stores absent-object pointers as ``0 * ptr + 1 * no_obj_ptr`` in
            # float32 (see the audit's runtime_state_probe), i.e. bit-equal to
            # the checkpoint anchor.
            anchor_mask = torch.eq(
                rows,
                source_anchor.to(device=rows.device, dtype=rows.dtype).reshape(-1),
            ).all(dim=-1, keepdim=True)
            flat_rows = rows.reshape(-1, rows.shape[-1])
            flat_mask = anchor_mask.reshape(-1)
            parameter = next(self.pointer_head.parameters())
            result = target_anchor.to(parameter.device, torch.float32).reshape(1, -1).expand(
                flat_rows.shape[0], -1
            ).clone()
            if bool((~flat_mask).any()):
                selected = flat_rows[~flat_mask]
                canonical = self.pointer_transform.source_to_canonical(selected)
                mapped = self.pointer_head(
                    canonical.to(parameter.device, parameter.dtype)
                )
                result[~flat_mask.to(parameter.device)] = (
                    self.pointer_transform.canonical_to_target(mapped)
                )
            return result.reshape(*rows.shape[:-1], result.shape[-1])
        canonical = self.pointer_transform.source_to_canonical(rows)
        parameter = next(self.pointer_head.parameters())
        mapped = self.pointer_head(canonical.to(parameter.device, parameter.dtype))
        return self.pointer_transform.canonical_to_target(mapped)

    def forward(self, source: CanonicalState) -> CanonicalState:
        source.validate()
        if source.spec != self.source_spec:
            raise ValueError(f"source spec {source.spec} does not match {self.source_spec}")
        spatial_rows = source.spatial_memory.float().movedim(3, -1)
        spatial = self.transform_spatial(spatial_rows).movedim(-1, 3)
        pointer = self.transform_pointer(source.object_pointer.float())
        return source.with_continuous(
            spatial_memory=spatial,
            object_pointer=pointer,
            presence_logits=source.presence_logits.clone(),
            positional_information={"policy": self.target_spec.positional_policy},
            translation_metadata={
                "translator": self.name,
                "architecture": self.architecture,
                "spatial_transform": self.spatial_transform.name,
                "pointer_transform": self.pointer_transform.name,
                "presence_logits": "diagnostic_only",
            },
        )

    translate = forward

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "cmmt.preconditioned_translator.v1",
            "source_spec": self.source_spec.to_dict(),
            "target_spec": self.target_spec.to_dict(),
            "architecture": self.architecture,
            "spatial_hidden_dim": self.spatial_hidden_dim,
            "pointer_hidden_dim": self.pointer_hidden_dim,
            "pointer_anchor_preserving": self.pointer_anchor_preserving,
            "fit_manifest_sha256": self.fit_manifest_sha256,
            "spatial_transform": self.spatial_transform.to_payload(),
            "pointer_transform": self.pointer_transform.to_payload(),
            "state_dict": {name: value.detach().cpu() for name, value in self.state_dict().items()},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PreconditionedStateTranslator":
        if payload.get("schema_version") != "cmmt.preconditioned_translator.v1":
            raise ValueError("unsupported preconditioned translator payload")
        translator = cls(
            StateSpec(**dict(payload["source_spec"])),
            StateSpec(**dict(payload["target_spec"])),
            FixedCanonicalTransform.from_payload(payload["spatial_transform"]),
            FixedCanonicalTransform.from_payload(payload["pointer_transform"]),
            architecture=str(payload["architecture"]),
            spatial_hidden_dim=int(payload["spatial_hidden_dim"]),
            pointer_hidden_dim=int(payload["pointer_hidden_dim"]),
            pointer_anchor_preserving=bool(payload["pointer_anchor_preserving"]),
            fit_manifest_sha256=payload.get("fit_manifest_sha256"),
        )
        translator.load_state_dict(dict(payload["state_dict"]), strict=True)
        translator.eval()
        return translator


def _record_index(pairs: Sequence[Pair]) -> list[tuple[int, int, int, int, str]]:
    records: list[tuple[int, int, int, int, str]] = []
    for pair_index, (source, target) in enumerate(pairs):
        valid = validate_paired_state_contract(source, target)
        dataset = str(source.metadata.get("dataset", target.metadata.get("dataset", "unknown")))
        for batch, obj, record in valid.nonzero(as_tuple=False).tolist():
            records.append((pair_index, batch, obj, record, dataset))
    if not records:
        raise ValueError("training pairs contain no valid records")
    return records


def fit_preconditioned_steps(
    translator: PreconditionedStateTranslator,
    pairs: Sequence[Pair],
    *,
    steps: int = 2000,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    batch_records: int = 16,
    spatial_positions_per_record: int = 512,
    seed: int = 7,
    device: str | torch.device = "cuda",
) -> list[dict[str, float]]:
    """Train independent heads with deterministic per-record spatial sampling."""

    if min(steps, batch_records, spatial_positions_per_record) < 1:
        raise ValueError("steps, batch_records and spatial_positions_per_record must be positive")
    records = _record_index(pairs)
    by_dataset: dict[str, list[tuple[int, int, int, int, str]]] = {}
    for record in records:
        by_dataset.setdefault(record[-1], []).append(record)
    datasets = sorted(by_dataset)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    translator.to(device=device, dtype=torch.float32)
    spatial_optimizer = torch.optim.Adam(
        translator.spatial_head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    pointer_optimizer = torch.optim.Adam(
        translator.pointer_head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[dict[str, float]] = []
    translator.train()
    for step in range(steps):
        chosen: list[tuple[int, int, int, int, str]] = []
        for offset in range(batch_records):
            dataset = datasets[(step * batch_records + offset) % len(datasets)]
            bucket = by_dataset[dataset]
            index = int(torch.randint(len(bucket), (1,), generator=generator))
            chosen.append(bucket[index])
        spatial_predictions: list[torch.Tensor] = []
        spatial_targets: list[torch.Tensor] = []
        pointer_predictions: list[torch.Tensor] = []
        pointer_targets: list[torch.Tensor] = []
        for pair_index, batch, obj, record, _dataset in chosen:
            source, target = pairs[pair_index]
            source_rows = source.spatial_memory[batch, obj, record].movedim(0, -1).reshape(
                -1, source.spec.feature_channels
            )
            target_rows = target.spatial_memory[batch, obj, record].movedim(0, -1).reshape(
                -1, target.spec.feature_channels
            )
            count = min(spatial_positions_per_record, source_rows.shape[0])
            positions = torch.randperm(source_rows.shape[0], generator=generator)[:count]
            spatial_predictions.append(translator.transform_spatial(source_rows[positions]))
            spatial_targets.append(target_rows[positions].to(device=device, dtype=torch.float32))
            pointer_predictions.append(
                translator.transform_pointer(source.object_pointer[batch, obj, record].view(1, -1))
            )
            pointer_targets.append(
                target.object_pointer[batch, obj, record].view(1, -1).to(device=device, dtype=torch.float32)
            )

        spatial_optimizer.zero_grad(set_to_none=True)
        spatial_loss = F.mse_loss(torch.cat(spatial_predictions), torch.cat(spatial_targets))
        if not torch.isfinite(spatial_loss):
            raise FloatingPointError(f"non-finite spatial loss at step {step}")
        spatial_loss.backward()
        spatial_optimizer.step()

        pointer_optimizer.zero_grad(set_to_none=True)
        pointer_loss = F.mse_loss(torch.cat(pointer_predictions), torch.cat(pointer_targets))
        if not torch.isfinite(pointer_loss):
            raise FloatingPointError(f"non-finite pointer loss at step {step}")
        pointer_loss.backward()
        pointer_optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            history.append(
                {
                    "step": float(step + 1),
                    "spatial_raw_mse": float(spatial_loss.detach().cpu()),
                    "pointer_raw_mse": float(pointer_loss.detach().cpu()),
                }
            )
    translator.eval()
    return history
