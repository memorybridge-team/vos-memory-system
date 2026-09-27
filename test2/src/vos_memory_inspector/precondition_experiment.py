"""Transform fitting, two-stage training, state evaluation, and report assembly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import torch

from .io_utils import atomic_torch_save, atomic_write_text, portable_path
from .metrics import evaluate_state
from .pair_bank import load_pair_bank
from .preconditioning import (
    FixedCanonicalTransform,
    Pair,
    PreconditionedStateTranslator,
    anchor_transform,
    consumer_metric_transform,
    fit_preconditioned_steps,
    fit_standardization,
    fit_zca,
    identity_transform,
    output_projection_svd_transform,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_banks(index_paths: Sequence[str | Path]) -> list[Pair]:
    pairs: list[Pair] = []
    for path in index_paths:
        pairs.extend(load_pair_bank(path))
    if not pairs:
        raise ValueError("pair-bank selection is empty")
    return pairs


_SPLITS = ("fit", "dev", "test")
_PAIR_TYPES = ("aligned", "native")


def _require_selection(
    pairs: Sequence[Pair], *, split: str, pair_type: str, role: str
) -> None:
    """Reject records whose bank identity differs from the requested selection.

    ``load_pair_bank`` stamps ``split`` and ``pair_type`` on both states of every
    record, so a mismatch here means the caller passed the wrong bank (for
    example dev/test records into transform fitting).
    """

    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
    if pair_type not in _PAIR_TYPES:
        raise ValueError(f"pair_type must be one of {_PAIR_TYPES}, got {pair_type!r}")
    if not pairs:
        raise ValueError(f"{role} pair selection is empty")
    expected = (split, pair_type)
    mismatched: list[tuple[int, tuple[Any, Any]]] = []
    for index, (source, target) in enumerate(pairs):
        for state in (source, target):
            actual = (state.metadata.get("split"), state.metadata.get("pair_type"))
            if actual != expected:
                mismatched.append((index, actual))
                break
    if mismatched:
        index, (actual_split, actual_type) = mismatched[0]
        missing = " (bank identity metadata missing)" if actual_split is None or actual_type is None else ""
        raise ValueError(
            f"{role} requires split={split!r} pair_type={pair_type!r} records; "
            f"{len(mismatched)} record(s) differ, first #{index} has "
            f"split={actual_split!r} pair_type={actual_type!r}{missing}"
        )


def _component_means(pairs: Sequence[Pair], component: str) -> tuple[torch.Tensor, torch.Tensor]:
    source_sum = None
    target_sum = None
    count = 0
    for source, target in pairs:
        valid = source.validity
        if component == "spatial_memory":
            source_rows = source.spatial_memory.movedim(3, -1)[valid].reshape(-1, source.spec.feature_channels).double()
            target_rows = target.spatial_memory.movedim(3, -1)[valid].reshape(-1, target.spec.feature_channels).double()
        else:
            source_rows = source.object_pointer[valid].reshape(-1, source.spec.pointer_dim).double()
            target_rows = target.object_pointer[valid].reshape(-1, target.spec.pointer_dim).double()
        source_sum = source_rows.sum(0) if source_sum is None else source_sum + source_rows.sum(0)
        target_sum = target_rows.sum(0) if target_sum is None else target_sum + target_rows.sum(0)
        count += source_rows.shape[0]
    if count == 0:
        raise ValueError("cannot calculate means from zero observations")
    return (source_sum / count).float(), (target_sum / count).float()


def fit_transform_suite(
    pairs: Sequence[Pair],
    *,
    checkpoint_matrices: str | Path,
    output: str | Path,
    fit_manifest_sha256: str,
    standardization_epsilon: float = 1e-5,
    zca_shrinkage: float = 1e-3,
    zca_eigenvalue_floor: float = 1e-4,
    consumer_regularization: float = 1e-5,
    svd_relative_floor: float = 1e-4,
) -> dict[str, Any]:
    """Fit all fixed transforms on aligned fit records only."""

    _require_selection(pairs, split="fit", pair_type="aligned", role="transform fitting")
    source_spec = pairs[0][0].spec
    target_spec = pairs[0][1].spec
    matrices = torch.load(checkpoint_matrices, map_location="cpu", weights_only=True)
    source_matrices = matrices["source"]
    target_matrices = matrices["target"]
    spatial_mean_s, spatial_mean_t = _component_means(pairs, "spatial_memory")
    pointer_mean_s, pointer_mean_t = _component_means(pairs, "object_pointer")
    raw_spatial = identity_transform(source_spec.feature_channels, target_spec.feature_channels)
    raw_pointer = identity_transform(source_spec.pointer_dim, target_spec.pointer_dim)
    spatial: dict[str, FixedCanonicalTransform] = {
        "M0_raw": raw_spatial,
        "M1_standardized": fit_standardization(
            pairs, "spatial_memory", epsilon=standardization_epsilon
        ),
        "M4_consumer": consumer_metric_transform(
            source_matrices["consumer_metric"],
            target_matrices["consumer_metric"],
            source_mean=spatial_mean_s,
            target_mean=spatial_mean_t,
            regularization=consumer_regularization,
        ),
        "M2_zca": fit_zca(
            pairs,
            "spatial_memory",
            shrinkage=zca_shrinkage,
            eigenvalue_floor=zca_eigenvalue_floor,
        ),
        "M3_out_proj_svd": output_projection_svd_transform(
            source_matrices["out_weight"],
            source_matrices["out_bias"],
            target_matrices["out_weight"],
            target_matrices["out_bias"],
            relative_floor=svd_relative_floor,
        ),
    }
    pointer_metric_source = torch.block_diag(*([source_matrices["consumer_metric"]] * 4))
    pointer_metric_target = torch.block_diag(*([target_matrices["consumer_metric"]] * 4))
    anchor_consumer = consumer_metric_transform(
        pointer_metric_source,
        pointer_metric_target,
        source_mean=source_matrices["no_obj_ptr"].reshape(-1),
        target_mean=target_matrices["no_obj_ptr"].reshape(-1),
        regularization=consumer_regularization,
    )
    anchor_consumer.settings.update(
        {
            "anchor_preserving_head_required": True,
            "source_anchor": source_matrices["no_obj_ptr"].reshape(-1),
            "target_anchor": target_matrices["no_obj_ptr"].reshape(-1),
        }
    )
    final_linear = output_projection_svd_transform(
        source_matrices["obj_ptr_final_weight"],
        source_matrices["obj_ptr_final_bias"],
        target_matrices["obj_ptr_final_weight"],
        target_matrices["obj_ptr_final_bias"],
        relative_floor=svd_relative_floor,
    )
    final_linear.settings.update(
        {
            "bypass_exact_anchor": True,
            "source_anchor": source_matrices["no_obj_ptr"].reshape(-1),
            "target_anchor": target_matrices["no_obj_ptr"].reshape(-1),
        }
    )
    pointer: dict[str, FixedCanonicalTransform] = {
        "P0_raw": raw_pointer,
        "P1_standardized": fit_standardization(
            pairs, "object_pointer", epsilon=standardization_epsilon
        ),
        "P2_anchor": anchor_transform(
            source_matrices["no_obj_ptr"], target_matrices["no_obj_ptr"]
        ),
        "P2_P4_anchor_consumer": anchor_consumer,
        "P3_final_linear": final_linear,
    }
    roundtrip = {"spatial": {}, "pointer": {}}
    spatial_sample = pairs[0][1].spatial_memory.float().movedim(3, -1)
    pointer_sample = pairs[0][1].object_pointer.float()
    for name, transform in spatial.items():
        roundtrip["spatial"][name] = transform.roundtrip_metrics(
            spatial_sample, deployment_dtype=torch.bfloat16
        )
    for name, transform in pointer.items():
        roundtrip["pointer"][name] = transform.roundtrip_metrics(
            pointer_sample, deployment_dtype=torch.float32
        )
    payload = {
        "schema_version": "cmmt.precondition_transform_suite.v1",
        "fit_manifest_sha256": fit_manifest_sha256,
        "source_spec": source_spec.to_dict(),
        "target_spec": target_spec.to_dict(),
        "spatial": {name: transform.to_payload() for name, transform in spatial.items()},
        "pointer": {name: transform.to_payload() for name, transform in pointer.items()},
        "roundtrip": roundtrip,
        "settings": {
            "standardization_epsilon": standardization_epsilon,
            "zca_shrinkage": zca_shrinkage,
            "zca_eigenvalue_floor": zca_eigenvalue_floor,
            "consumer_regularization": consumer_regularization,
            "svd_relative_floor": svd_relative_floor,
            "statistics_scope": "aligned_fit_only",
        },
    }
    output = Path(output).resolve()
    atomic_torch_save(payload, output)
    atomic_write_text(
        output.with_suffix(output.suffix + ".sha256"),
        f"{sha256_file(output)}  {output.name}\n",
        encoding="ascii",
    )
    return payload


def _load_transform_suite(path: str | Path) -> Mapping[str, Any]:
    path = Path(path).resolve()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").split()[0] != sha256_file(path):
        raise ValueError("transform suite checksum is missing or invalid")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "cmmt.precondition_transform_suite.v1":
        raise ValueError("unsupported transform suite")
    return payload


def train_two_stage(
    *,
    transform_suite: str | Path,
    aligned_pairs: Sequence[Pair],
    native_pairs: Sequence[Pair],
    spatial_method: str,
    pointer_method: str,
    architecture: str,
    output_dir: str | Path,
    seed: int = 7,
    device: str = "cuda",
    aligned_steps: int = 2000,
    native_steps: int = 2000,
    aligned_learning_rate: float = 1e-3,
    native_learning_rate: float = 1e-4,
    batch_records: int = 16,
    spatial_positions_per_record: int = 512,
) -> dict[str, Any]:
    _require_selection(aligned_pairs, split="fit", pair_type="aligned", role="aligned training")
    _require_selection(native_pairs, split="fit", pair_type="native", role="native fine-tuning")
    suite = _load_transform_suite(transform_suite)
    try:
        spatial_transform = FixedCanonicalTransform.from_payload(suite["spatial"][spatial_method])
        pointer_transform = FixedCanonicalTransform.from_payload(suite["pointer"][pointer_method])
    except KeyError as exc:
        raise ValueError(f"unknown transform method: {exc}") from exc
    # Seed before constructing the heads: their initial weights consume the
    # global RNG, so seeding later would make ``seed`` control only sampling.
    torch.manual_seed(seed)
    translator = PreconditionedStateTranslator(
        aligned_pairs[0][0].spec,
        aligned_pairs[0][1].spec,
        spatial_transform,
        pointer_transform,
        architecture=architecture,
        spatial_hidden_dim=128,
        pointer_hidden_dim=512,
        pointer_anchor_preserving=pointer_method in {
            "P2_anchor",
            "P2_P4_anchor_consumer",
        },
        fit_manifest_sha256=str(suite["fit_manifest_sha256"]),
    )
    aligned_history = fit_preconditioned_steps(
        translator,
        aligned_pairs,
        steps=aligned_steps,
        learning_rate=aligned_learning_rate,
        batch_records=batch_records,
        spatial_positions_per_record=spatial_positions_per_record,
        seed=seed,
        device=device,
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = output_dir / "translator_aligned.pt"
    atomic_torch_save(translator.to_payload(), aligned_path)
    native_history = []
    native_path = output_dir / "translator_native_finetuned.pt"
    if native_steps > 0:
        native_history = fit_preconditioned_steps(
            translator,
            native_pairs,
            steps=native_steps,
            learning_rate=native_learning_rate,
            batch_records=batch_records,
            spatial_positions_per_record=spatial_positions_per_record,
            seed=seed,
            device=device,
        )
        atomic_torch_save(translator.to_payload(), native_path)
    report = {
        "schema_version": "cmmt.two_stage_training.v1",
        "spatial_method": spatial_method,
        "pointer_method": pointer_method,
        "architecture": architecture,
        "seed": seed,
        "training_dtype": "float32",
        "backbone_frozen": True,
        "sampling": {
            "batch_records": batch_records,
            "spatial_positions_per_record": spatial_positions_per_record,
            "dataset_balance": "equal round-robin across dataset labels",
        },
        "aligned": {
            "steps": aligned_steps,
            "learning_rate": aligned_learning_rate,
            "history": aligned_history,
            "artifact": portable_path(aligned_path),
            "artifact_sha256": sha256_file(aligned_path),
        },
        "native_finetune": {
            "steps": native_steps,
            "learning_rate": native_learning_rate,
            "optimizer_reinitialized": True,
            "fixed_transforms_reused": True,
            "history": native_history,
            "artifact": portable_path(native_path) if native_steps > 0 else None,
            "artifact_sha256": sha256_file(native_path) if native_steps > 0 else None,
        },
    }
    atomic_write_text(
        output_dir / "training_report.json", json.dumps(report, indent=2, ensure_ascii=False)
    )
    return report


def evaluate_translator_state(
    artifact: str | Path,
    pairs: Sequence[Pair],
    *,
    output: str | Path,
    pair_type: str,
    split: str,
) -> dict[str, Any]:
    _require_selection(pairs, split=split, pair_type=pair_type, role="state evaluation")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    translator = PreconditionedStateTranslator.from_payload(payload)
    rows = []
    with torch.no_grad():
        for source, target in pairs:
            evaluation = evaluate_state(translator.translate(source), target)
            evaluation["dataset"] = str(source.metadata.get("dataset", "unknown"))
            evaluation["video_id"] = str(source.metadata.get("video_id", "unknown"))
            rows.append(evaluation)
    metric_paths = {
        "aggregate_mse": lambda row: row["aggregate_mse"],
        "spatial_mse": lambda row: row["components"]["spatial_memory"]["mse"],
        "spatial_cosine": lambda row: row["components"]["spatial_memory"]["cosine"],
        "pointer_mse": lambda row: row["components"]["object_pointer"]["mse"],
        "pointer_cosine": lambda row: row["components"]["object_pointer"]["cosine"],
    }
    videos: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        videos.setdefault((row["dataset"], row["video_id"]), []).append(row)
    video_means: dict[tuple[str, str], dict[str, float]] = {
        key: {name: mean(accessor(row) for row in group) for name, accessor in metric_paths.items()}
        for key, group in videos.items()
    }
    datasets: dict[str, list[dict[str, float]]] = {}
    for (dataset, _video), metrics in video_means.items():
        datasets.setdefault(dataset, []).append(metrics)
    dataset_means = {
        dataset: {
            name: mean(video[name] for video in dataset_videos)
            for name in metric_paths
        }
        for dataset, dataset_videos in datasets.items()
    }
    balanced = {
        name: mean(dataset[name] for dataset in dataset_means.values())
        for name in metric_paths
    }
    report = {
        "schema_version": "cmmt.precondition_state_evaluation.v1",
        "artifact": portable_path(artifact),
        "artifact_sha256": sha256_file(artifact),
        "pair_type": pair_type,
        "split": split,
        "record_count": len(rows),
        **balanced,
        "aggregation": "record mean within video; video mean within dataset; equal dataset mean",
        "dataset_means": dataset_means,
        "video_means": {
            f"{dataset}/{video}": metrics
            for (dataset, video), metrics in video_means.items()
        },
        "rows": rows,
        "scope": "state reconstruction only; not a no-replay VOS score",
    }
    output = Path(output).resolve()
    atomic_write_text(output, json.dumps(report, indent=2, ensure_ascii=False))
    return report


def assemble_report(experiment_root: str | Path) -> Path:
    root = Path(experiment_root).resolve()
    evaluations = []
    for path in sorted((root / "evaluations").glob("*.json")) if (root / "evaluations").is_dir() else []:
        try:
            evaluations.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
        except (ValueError, OSError):
            continue
    audit_path = root / "checkpoint_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
    transform_path = root / "roundtrip" / "transform_suite.pt"
    transform = (
        torch.load(transform_path, map_location="cpu", weights_only=True)
        if transform_path.is_file()
        else None
    )
    aligned_evaluations = [
        (name, report) for name, report in evaluations if report.get("pair_type") == "aligned"
    ]
    native_evaluations = [
        (name, report) for name, report in evaluations if report.get("pair_type") == "native"
    ]
    continuation_evaluations = [
        (name, report) for name, report in evaluations if "candidate_j_and_f" in report
    ]
    lines = [
        "# SAM 2.1 Small → Base+ Memory Preconditioning Report",
        "",
        "## A. Environment",
        "",
    ]
    if audit is None:
        lines.append("Checkpoint audit has not been run.")
    else:
        environment = audit["environment"]
        lines.extend(
            [
                f"- SAM2 upstream commit: `{environment['git_commit']}`",
                f"- PyTorch/CUDA: `{environment['pytorch']}` / `{environment['cuda_runtime']}`",
                f"- GPU: `{environment['gpu']}`",
                f"- Inference/translator dtype: `{environment['inference_dtype']}` / `{environment['translator_training_dtype']}`",
                f"- Small checkpoint SHA-256: `{audit['inputs']['source_checkpoint_sha256']}`",
                f"- Base+ checkpoint SHA-256: `{audit['inputs']['target_checkpoint_sha256']}`",
            ]
        )
    lines.extend(
        [
        "",
        "## B. Verified SAM2.1 memory path",
        "",
        "See `code_audit.md`. Target positional encodings and temporal assembly remain target-native.",
        "",
        "## C. Checkpoint audit",
        "",
        "See `checkpoint_audit.md`, `checkpoint_audit.json`, and `checkpoint_matrices.pt`.",
        "",
        "## D. Candidate transforms",
        "",
        "Spatial: M0 raw, M1 standardization, M4 consumer-aware, M2 ZCA, M3 output-projection SVD. "
        "Pointer: P0 raw, P1 standardization, P2 anchor, P2+P4 anchor/consumer, P3 final-linear.",
        "",
        "## E. Round-trip sanity",
        "",
        ]
    )
    if transform is None:
        lines.append("Transform suite has not been fit.")
    else:
        for component in ("spatial", "pointer"):
            for name, metrics in transform["roundtrip"][component].items():
                lines.append(
                    f"- {component}/{name}: relative L2={metrics['relative_l2']:.3e}, "
                    f"finite={metrics['finite']}, deployment finite={metrics['deployment_finite']}."
                )
    lines.extend(["", "## F. Aligned-pair representation results", ""])
    if aligned_evaluations:
        for name, report in aligned_evaluations:
            lines.append(
                f"- `{name}`: records={report.get('record_count', 'n/a')}, "
                f"spatial MSE={report.get('spatial_mse', 'n/a')}, "
                f"pointer MSE={report.get('pointer_mse', 'n/a')}."
            )
    else:
        lines.append("No aligned evaluation has completed.")
    lines.extend(["", "## G. Native-pair representation results", ""])
    if native_evaluations:
        for name, report in native_evaluations:
            lines.append(
                f"- `{name}`: records={report.get('record_count', 'n/a')}, "
                f"spatial MSE={report.get('spatial_mse', 'n/a')}, "
                f"pointer MSE={report.get('pointer_mse', 'n/a')}."
            )
    else:
        lines.append("No native evaluation has completed.")
    lines.extend(
        [
            "",
            "## H. Consumer/attention results",
            "",
            "The same-query K/V and attention diagnostic is implemented; held-out results are pending.",
            "",
            "## I. Actual no-replay continuation",
            "",
        ]
    )
    if continuation_evaluations:
        for name, report in continuation_evaluations:
            score = report["candidate_j_and_f"]["mean_J_and_F"]
            lines.append(f"- `{name}`: suffix J&F={score:.6f}.")
    else:
        lines.append("Held-out no-replay continuation has not been run; no method conclusion is made.")
    lines.extend(
        [
            "",
            "## J. Runtime and transfer cost",
            "",
            "Pending held-out no-replay runs. Reference generation and target initialization are reported separately by the evaluator.",
            "",
            "## K. Interpretation",
            "",
            "The completed numbers are one-video fit sanity checks. They show that the pipeline trains, "
            "but they do not compare canonicalization candidates or establish VOS quality. The final "
            "decision remains held-out native Small-prefix → Base+ no-replay suffix performance; lower "
            "state error alone is not treated as success.",
            "",
        ]
    )
    path = root / "report.md"
    atomic_write_text(path, "\n".join(lines))
    return path
