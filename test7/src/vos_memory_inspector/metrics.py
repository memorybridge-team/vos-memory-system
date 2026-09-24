"""Tensor and systems metrics for paired handoff states."""

from __future__ import annotations

import time
from statistics import median
from typing import Any, Callable

import torch
from torch.nn import functional as F

from .state_schema import CanonicalState


def _valid_rows(
    prediction: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = validity.to(prediction.device)
    pred = prediction[mask].reshape(int(mask.sum().item()), -1)
    truth = target.to(prediction.device)[mask].reshape(int(mask.sum().item()), -1)
    return pred.float(), truth.float()


def tensor_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor,
) -> dict[str, float]:
    pred, truth = _valid_rows(prediction, target, validity)
    difference = pred - truth
    mse = difference.square().mean()
    cosine = F.cosine_similarity(pred, truth, dim=-1, eps=1e-12).mean()
    relative = difference.norm(dim=-1) / truth.norm(dim=-1).clamp_min(1e-12)
    return {
        "mse": float(mse.cpu()),
        "cosine": float(cosine.cpu()),
        "relative_error": float(relative.mean().cpu()),
    }


def evaluate_state(
    prediction: CanonicalState,
    target: CanonicalState,
) -> dict[str, Any]:
    prediction.validate()
    target.validate()
    if prediction.spatial_memory.shape != target.spatial_memory.shape:
        raise ValueError("prediction and target spatial shapes differ")
    validity = prediction.validity & target.validity.to(prediction.validity.device)
    components = {
        "spatial_memory": tensor_metrics(
            prediction.spatial_memory, target.spatial_memory, validity
        ),
        "object_pointer": tensor_metrics(
            prediction.object_pointer, target.object_pointer, validity
        ),
    }
    aggregate_mse = sum(value["mse"] for value in components.values()) / len(
        components
    )
    return {
        "components": components,
        "diagnostics": {
            "presence_logits": tensor_metrics(
                prediction.presence_logits, target.presence_logits, validity
            )
        },
        "aggregate_mse": aggregate_mse,
        "valid_records": int(validity.sum().item()),
        "translated_bytes": prediction.handoff_bytes(),
    }


def benchmark_translation(
    operation: Callable[[], CanonicalState],
    *,
    warmup: int = 3,
    repeats: int = 10,
) -> dict[str, float]:
    with torch.no_grad():
        for _ in range(warmup):
            result = operation()
        if result.spatial_memory.is_cuda:
            torch.cuda.synchronize(result.spatial_memory.device)
        timings: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = operation()
            if result.spatial_memory.is_cuda:
                torch.cuda.synchronize(result.spatial_memory.device)
            timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "repeats": float(repeats),
    }


def direct_improvement(model_mse: float, direct_mse: float) -> float:
    if direct_mse <= 0:
        return 0.0
    return (direct_mse - model_mse) / direct_mse
