"""Losses and optimizer loop for learned canonical-state translators."""

from __future__ import annotations

from typing import Iterable

import torch
from torch.nn import functional as F

from .device import resolve_device
from .state_schema import CanonicalState
from .translators import _LearnedStateTranslator, _pair_guard, _resample_spatial


def state_mse_loss(prediction: CanonicalState, target: CanonicalState) -> torch.Tensor:
    valid = _pair_guard(prediction, target)
    target_device = prediction.spatial_memory.device
    valid = valid.to(target_device)
    spatial_error = (prediction.spatial_memory - target.spatial_memory.to(target_device)) ** 2
    spatial_loss = spatial_error[valid].mean()
    pointer_loss = (
        (prediction.object_pointer - target.object_pointer.to(prediction.object_pointer.device))
        ** 2
    )[valid.to(prediction.object_pointer.device)].mean()
    return spatial_loss + pointer_loss


def fit_gradient_translator(
    translator: _LearnedStateTranslator,
    pairs: Iterable[tuple[CanonicalState, CanonicalState]],
    *,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str | torch.device | None = None,
    spatial_samples_per_pair: int | None = None,
) -> list[float]:
    device = resolve_device(device)
    pair_list = list(pairs)
    if not pair_list:
        raise ValueError("at least one paired state is required")
    if spatial_samples_per_pair is not None and spatial_samples_per_pair < 1:
        raise ValueError("spatial_samples_per_pair must be positive")
    translator.to(device)
    optimizer = torch.optim.Adam(translator.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(torch.initial_seed())
    history: list[float] = []
    translator.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        if spatial_samples_per_pair is None:
            losses = [state_mse_loss(translator(source), target) for source, target in pair_list]
        else:
            losses = [
                _sampled_state_mse_loss(
                    translator,
                    source,
                    target,
                    spatial_samples=spatial_samples_per_pair,
                    generator=generator,
                )
                for source, target in pair_list
            ]
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    translator.eval()
    return history


def _sampled_state_mse_loss(
    translator: _LearnedStateTranslator,
    source: CanonicalState,
    target: CanonicalState,
    *,
    spatial_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Estimate component-balanced state loss without materializing full MLP output."""

    valid = _pair_guard(source, target)
    source_spatial = _resample_spatial(
        source.spatial_memory,
        translator.target_spec.height,
        translator.target_spec.width,
    )
    source_vectors = source_spatial.movedim(3, -1)[valid].reshape(
        -1, translator.source_spec.feature_channels
    )
    target_vectors = target.spatial_memory.movedim(3, -1)[valid].reshape(
        -1, translator.target_spec.feature_channels
    )
    sample_count = min(spatial_samples, source_vectors.shape[0])
    indices = torch.randperm(
        source_vectors.shape[0], generator=generator
    )[:sample_count]
    predicted_spatial = translator._feature_head(source_vectors[indices])
    spatial_loss = F.mse_loss(
        predicted_spatial,
        target_vectors[indices].to(
            device=predicted_spatial.device, dtype=predicted_spatial.dtype
        ),
    )

    source_pointer = source.object_pointer[valid]
    predicted_pointer = translator._pointer_head(source_pointer)
    pointer_loss = F.mse_loss(
        predicted_pointer,
        target.object_pointer[valid].to(
            device=predicted_pointer.device, dtype=predicted_pointer.dtype
        ),
    )
    return spatial_loss + pointer_loss
