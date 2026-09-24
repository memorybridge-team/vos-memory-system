"""Paired-state experiment utilities and an explicitly synthetic smoke run."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.nn import functional as F

from .case_cache import load_case_cache
from .device import resolve_device
from .metrics import benchmark_translation, direct_improvement, evaluate_state
from .state_schema import CanonicalState, StateSpec
from .translator_training import fit_gradient_translator
from .translators import (
    DirectCopyTranslator,
    LinearStateTranslator,
    ResidualMLPStateTranslator,
    RidgeStateTranslator,
)


def _source_state(generator: torch.Generator, *, switch_frame: int) -> CanonicalState:
    batch, objects, records, channels, height, width = 1, 2, 4, 6, 4, 5
    spatial = torch.randn(
        batch, objects, records, channels, height, width, generator=generator
    )
    pointer = torch.randn(batch, objects, records, 8, generator=generator)
    presence = torch.randn(batch, objects, records, 1, generator=generator)
    frame_indices = torch.tensor([[[0, 2, 4, 6], [0, 2, 4, 6]]])
    slot_order = torch.arange(records).view(1, 1, records).expand(batch, objects, records).clone()
    is_conditioning = torch.tensor([[[True, False, False, False]]]).expand(
        batch, objects, records
    ).clone()
    validity = torch.ones(batch, objects, records, dtype=torch.bool)
    return CanonicalState(
        spatial_memory=spatial,
        object_pointer=pointer,
        presence_logits=presence,
        frame_indices=frame_indices,
        slot_order=slot_order,
        is_conditioning=is_conditioning,
        validity=validity,
        object_ids=("synthetic-object-0", "synthetic-object-1"),
        switch_frame=switch_frame,
        positional_information={"policy": "regenerate_at_target"},
        metadata={"synthetic": True},
    ).validate()


def _fixed_relation(seed: int = 913) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "feature_weight": torch.randn(6, 7, generator=generator) / 2,
        "feature_curve": torch.randn(6, 7, generator=generator) / 12,
        "feature_bias": torch.randn(7, generator=generator) / 4,
        "pointer_weight": torch.randn(8, 5, generator=generator) / 2,
        "pointer_curve": torch.randn(8, 5, generator=generator) / 15,
        "pointer_bias": torch.randn(5, generator=generator) / 4,
        "presence_weight": torch.tensor([[1.35]]),
        "presence_bias": torch.tensor([-0.2]),
    }


def _target_state(source: CanonicalState, relation: dict[str, torch.Tensor]) -> CanonicalState:
    batch, objects, records, channels, old_h, old_w = source.spatial_memory.shape
    resized = F.interpolate(
        source.spatial_memory.reshape(batch * objects * records, channels, old_h, old_w),
        size=(6, 4),
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, objects, records, channels, 6, 4)
    features = resized.movedim(3, -1)
    target_feature = (
        features @ relation["feature_weight"]
        + features.square() @ relation["feature_curve"]
        + relation["feature_bias"]
    ).movedim(-1, 3)
    pointer = (
        source.object_pointer @ relation["pointer_weight"]
        + source.object_pointer.square() @ relation["pointer_curve"]
        + relation["pointer_bias"]
    )
    presence = (
        source.presence_logits @ relation["presence_weight"]
        + relation["presence_bias"]
    )
    return source.with_continuous(
        spatial_memory=target_feature,
        object_pointer=pointer,
        presence_logits=presence,
        positional_information={"policy": "provided_target", "synthetic": True},
        translation_metadata={"ground_truth": "fixed_synthetic_nonlinear_relation"},
    )


def make_synthetic_pairs(
    *, seed: int, train_count: int = 6, test_count: int = 3
) -> tuple[
    list[tuple[CanonicalState, CanonicalState]],
    list[tuple[CanonicalState, CanonicalState]],
]:
    generator = torch.Generator().manual_seed(seed)
    relation = _fixed_relation()
    pairs = []
    for index in range(train_count + test_count):
        source = _source_state(generator, switch_frame=6 + index)
        pairs.append((source, _target_state(source, relation)))
    return pairs[:train_count], pairs[train_count:]


def load_canonical_state(path: Path) -> CanonicalState:
    """Load an explicitly trusted canonical-state artifact."""

    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, CanonicalState):
        return value.validate()
    if isinstance(value, dict):
        candidates = [item for item in value.values() if isinstance(item, CanonicalState)]
        if len(candidates) == 1:
            return candidates[0].validate()
    raise TypeError(f"{path} does not contain exactly one CanonicalState")


def load_case_cache_pair(path: Path) -> tuple[CanonicalState, CanonicalState]:
    """Load an aligned source/target pair from a checksummed prepared case."""

    payload = load_case_cache(path)
    source = payload["source_canonical"].validate()
    target = payload["target_canonical"].validate()
    return source, target


def _mean_evaluation(
    translator: Any, pairs: list[tuple[CanonicalState, CanonicalState]]
) -> dict[str, Any]:
    evaluations = []
    with torch.no_grad():
        for source, target in pairs:
            evaluations.append(evaluate_state(translator.translate(source), target))
    components: dict[str, dict[str, float]] = {}
    for component in evaluations[0]["components"]:
        components[component] = {
            metric: mean(item["components"][component][metric] for item in evaluations)
            for metric in ("mse", "cosine", "relative_error")
        }
    return {
        "components": components,
        "aggregate_mse": mean(item["aggregate_mse"] for item in evaluations),
        "valid_records": sum(item["valid_records"] for item in evaluations),
        "translated_bytes_per_state": evaluations[0]["translated_bytes"],
    }


def run_synthetic_experiment(
    output_dir: Path,
    *,
    seed: int = 7,
    epochs: int = 120,
    learning_rate: float = 2e-2,
    ridge_lambda: float = 0.01,
    hidden_dim: int = 32,
) -> dict[str, Any]:
    """Run a deterministic CPU smoke test; this is not a SAM 2 result."""

    torch.manual_seed(seed)
    train_pairs, test_pairs = make_synthetic_pairs(seed=seed)
    source_spec = StateSpec.from_state(train_pairs[0][0])
    target_spec = StateSpec.from_state(train_pairs[0][1])

    direct = DirectCopyTranslator(target_spec)
    ridge = RidgeStateTranslator.fit(train_pairs, ridge_lambda=ridge_lambda)
    linear = LinearStateTranslator(source_spec, target_spec)
    mlp = ResidualMLPStateTranslator(
        source_spec, target_spec, hidden_dim=hidden_dim
    )
    linear_history = fit_gradient_translator(
        linear, train_pairs, epochs=epochs, learning_rate=learning_rate
    )
    mlp_history = fit_gradient_translator(
        mlp, train_pairs, epochs=epochs, learning_rate=learning_rate
    )
    translators = {
        "direct": direct,
        "ridge": ridge,
        "linear": linear,
        "residual_mlp": mlp,
    }
    results: dict[str, Any] = {}
    direct_mse = _mean_evaluation(direct, test_pairs)["aggregate_mse"]
    latency_source = test_pairs[0][0]
    for name, translator in translators.items():
        evaluation = _mean_evaluation(translator, test_pairs)
        evaluation["parameter_count"] = translator.parameter_count()
        evaluation["latency_cpu"] = benchmark_translation(
            lambda translator=translator: translator.translate(latency_source),
            warmup=2,
            repeats=8,
        )
        evaluation["direct_mse_improvement"] = direct_improvement(
            evaluation["aggregate_mse"], direct_mse
        )
        results[name] = evaluation

    # A downstream-shaped deterministic readout catches gross state/axis errors.
    # It is intentionally labelled synthetic and is not a SAM 2 injection test.
    with torch.no_grad():
        predicted = mlp.translate(test_pairs[0][0])
        oracle = test_pairs[0][1]
        predicted_readout = (
            predicted.spatial_memory.mean()
            + predicted.object_pointer.mean()
        )
        oracle_readout = (
            oracle.spatial_memory.mean()
            + oracle.object_pointer.mean()
        )
        continuation_smoke = {
            "kind": "synthetic_readout_only",
            "passed": bool(torch.isfinite(predicted_readout)),
            "absolute_error": float((predicted_readout - oracle_readout).abs()),
        }

    report = {
        "schema_version": "cmmt.paired_experiment.v1",
        "synthetic": True,
        "warning": "Synthetic smoke only; no SAM 2 checkpoint or video was used.",
        "seed": seed,
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "source_spec": source_spec.to_dict(),
        "target_spec": target_spec.to_dict(),
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "ridge_lambda": ridge_lambda,
            "linear_initial_loss": linear_history[0],
            "linear_final_loss": linear_history[-1],
            "mlp_initial_loss": mlp_history[0],
            "mlp_final_loss": mlp_history[-1],
        },
        "results": results,
        "continuation_smoke": continuation_smoke,
        "actual_sam2_injection": {
            "ran": False,
            "reason": "No local source/target SAM 2 checkpoints were provided.",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "synthetic_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save(
        {"b_t": test_pairs[0][0], "a_t": test_pairs[0][1], "synthetic": True},
        output_dir / "synthetic_pair.pt",
    )
    torch.save(
        {"linear": linear.state_dict(), "residual_mlp": mlp.state_dict()},
        output_dir / "synthetic_translators.pt",
    )
    _write_markdown_report(report, output_dir / "synthetic_report.md")
    return report


def run_paired_experiment(
    train_pairs: list[tuple[CanonicalState, CanonicalState]],
    test_pairs: list[tuple[CanonicalState, CanonicalState]],
    output_dir: Path,
    *,
    seed: int = 7,
    epochs: int = 120,
    learning_rate: float = 2e-2,
    ridge_lambda: float = 0.01,
    hidden_dim: int = 128,
    translator_names: tuple[str, ...] | None = None,
    device: str | None = None,
    spatial_samples_per_pair: int | None = None,
) -> dict[str, Any]:
    """Fit/evaluate offline paired canonical state without claiming injection."""

    if not train_pairs or not test_pairs:
        raise ValueError("separate non-empty train_pairs and test_pairs are required")
    allowed = ("direct", "ridge", "linear", "residual_mlp")
    selected = allowed if translator_names is None else tuple(dict.fromkeys(translator_names))
    if not selected:
        raise ValueError("at least one translator must be selected")
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise ValueError(f"unknown translators: {invalid}; allowed={list(allowed)}")
    device = resolve_device(device)
    torch.manual_seed(seed)
    source_spec = train_pairs[0][0].spec
    target_spec = train_pairs[0][1].spec
    direct = DirectCopyTranslator(target_spec)
    translators: dict[str, Any] = {}
    training: dict[str, Any] = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "ridge_lambda": ridge_lambda,
        "device": device,
        "spatial_samples_per_pair": spatial_samples_per_pair,
    }
    if "direct" in selected:
        translators["direct"] = direct
    if "ridge" in selected:
        translators["ridge"] = RidgeStateTranslator.fit(
            train_pairs, ridge_lambda=ridge_lambda
        )
    if "linear" in selected:
        linear = LinearStateTranslator(source_spec, target_spec)
        linear_history = fit_gradient_translator(
            linear,
            train_pairs,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            spatial_samples_per_pair=spatial_samples_per_pair,
        )
        translators["linear"] = linear
        training["linear_initial_loss"] = linear_history[0]
        training["linear_final_loss"] = linear_history[-1]
    if "residual_mlp" in selected:
        mlp = ResidualMLPStateTranslator(
            source_spec, target_spec, hidden_dim=hidden_dim
        )
        mlp_history = fit_gradient_translator(
            mlp,
            train_pairs,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            spatial_samples_per_pair=spatial_samples_per_pair,
        )
        translators["residual_mlp"] = mlp
        training["mlp_initial_loss"] = mlp_history[0]
        training["mlp_final_loss"] = mlp_history[-1]
    direct_mse = _mean_evaluation(direct, test_pairs)["aggregate_mse"]
    results: dict[str, Any] = {}
    for name, translator in translators.items():
        evaluation = _mean_evaluation(translator, test_pairs)
        evaluation["parameter_count"] = translator.parameter_count()
        evaluation["latency_cpu"] = benchmark_translation(
            lambda translator=translator: translator.translate(test_pairs[0][0]),
            warmup=2,
            repeats=8,
        )
        evaluation["direct_mse_improvement"] = direct_improvement(
            evaluation["aggregate_mse"], direct_mse
        )
        results[name] = evaluation
    report = {
        "schema_version": "cmmt.paired_experiment.v1",
        "synthetic": False,
        "warning": (
            "Offline paired-state result only; target next-frame injection and "
            "downstream video quality are not evaluated by this command."
        ),
        "seed": seed,
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "translator_names": list(selected),
        "source_spec": source_spec.to_dict(),
        "target_spec": target_spec.to_dict(),
        "training": training,
        "results": results,
        "actual_sam2_injection": {
            "ran": False,
            "reason": "This command evaluates serialized states only.",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    serialized: dict[str, Any] = {}
    ridge = translators.get("ridge")
    if ridge is not None:
        serialized["ridge"] = ridge.to_payload()
    linear = translators.get("linear")
    if linear is not None:
        serialized["linear"] = linear.state_dict()
    mlp = translators.get("residual_mlp")
    if mlp is not None:
        serialized["residual_mlp"] = mlp.to_payload()
    torch.save(serialized, output_dir / "paired_translators.pt")
    _write_markdown_report(report, output_dir / "paired_report.md")
    return report


def _write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        (
            "# Synthetic paired-state smoke result"
            if report["synthetic"]
            else "# Paired-state translator result"
        ),
        "",
        f"> {report['warning']}",
        "",
        "| Translator | Aggregate MSE | Direct improvement | Parameters | Median CPU ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in report["results"].items():
        lines.append(
            f"| {name} | {result['aggregate_mse']:.6g} | "
            f"{result['direct_mse_improvement']:.2%} | {result['parameter_count']} | "
            f"{result['latency_cpu']['median_ms']:.4f} |"
        )
    lines.extend(
        [
            "",
            "This report evaluates serialized tensor state only. It is not an "
            "actual SAM 2 next-frame injection result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
