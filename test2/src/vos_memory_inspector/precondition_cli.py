"""Stage-oriented CLI for the Small → Base+ preconditioning experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .artifacts import sam2_frame_paths
from .checkpoint_audit import run_checkpoint_audit
from .davis_evaluation import evaluate_davis_future_masks, load_official_davis_metrics
from .data_acquisition import SOURCES, acquire_dataset
from .io_utils import atomic_write_text
from .precondition_experiment import (
    assemble_report,
    evaluate_translator_state,
    fit_transform_suite,
    load_banks,
    train_two_stage,
)
from .precondition_manifest import build_train_split_manifest, write_train_split_manifest
from .preconditioning import PreconditionedStateTranslator
from .roundtrip import run_cached_translator_handoff
from .temporal_evaluation import evaluate_temporal_handoff
from .state_pair_extraction import extract_pair_bank


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAM 2.1 Small → Base+ memory preconditioning experiment"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    download = subcommands.add_parser("download", help="acquire an official dataset archive")
    download.add_argument("source", choices=sorted(SOURCES))
    download.add_argument("--download-dir", required=True, type=Path)
    download.add_argument("--extract-dir", required=True, type=Path)
    download.add_argument(
        "--allow-checksum-mismatch",
        action="store_true",
        help="record but do not reject an archive whose SHA-256 differs from the pinned value",
    )

    audit = subcommands.add_parser("audit", help="audit pinned source and checkpoint matrices")
    audit.add_argument("--sam2-repo", required=True, type=Path)
    audit.add_argument("--source-config", required=True)
    audit.add_argument("--source-checkpoint", required=True, type=Path)
    audit.add_argument("--target-config", required=True)
    audit.add_argument("--target-checkpoint", required=True, type=Path)
    audit.add_argument("--output-dir", required=True, type=Path)
    audit.add_argument("--device", default="cpu")

    split = subcommands.add_parser("split", help="freeze an 80/10/10 video-level train split")
    split.add_argument("--dataset-root", required=True, type=Path)
    split.add_argument("--dataset", required=True, choices=("LVOS_V2", "VOST"))
    split.add_argument("--release", required=True)
    split.add_argument("--official-split", default="train")
    split.add_argument("--seed", type=int, default=7)
    split.add_argument("--output", required=True, type=Path)

    extract = subcommands.add_parser("extract", help="extract aligned/native pair banks")
    extract.add_argument("--manifest", required=True, type=Path)
    extract.add_argument("--dataset-root", required=True, type=Path)
    extract.add_argument("--output-root", required=True, type=Path)
    extract.add_argument("--sam2-repo", required=True, type=Path)
    extract.add_argument("--source-config", required=True)
    extract.add_argument("--source-checkpoint", required=True, type=Path)
    extract.add_argument("--target-config", required=True)
    extract.add_argument("--target-checkpoint", required=True, type=Path)
    extract.add_argument("--pair-type", action="append", choices=("aligned", "native"))
    extract.add_argument("--full-train", action="store_true")
    extract.add_argument("--tiny-overfit", action="store_true")
    extract.add_argument("--device", default="cuda")

    transforms = subcommands.add_parser("fit-transforms", help="fit fixed transforms on aligned fit only")
    transforms.add_argument("--aligned-fit-bank", required=True, action="append", type=Path)
    transforms.add_argument("--checkpoint-matrices", required=True, type=Path)
    transforms.add_argument("--fit-manifest", required=True, type=Path)
    transforms.add_argument("--output", required=True, type=Path)

    train = subcommands.add_parser("train", help="aligned training followed by native fine-tuning")
    train.add_argument("--transform-suite", required=True, type=Path)
    train.add_argument("--aligned-bank", required=True, action="append", type=Path)
    train.add_argument("--native-bank", required=True, action="append", type=Path)
    train.add_argument("--spatial-method", required=True)
    train.add_argument("--pointer-method", required=True)
    train.add_argument("--architecture", choices=("linear", "mlp"), default="mlp")
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--device", default="cuda")
    train.add_argument("--aligned-steps", type=int, default=2000)
    train.add_argument("--native-steps", type=int, default=2000)
    train.add_argument("--aligned-lr", type=float, default=1e-3)
    train.add_argument("--native-lr", type=float, default=1e-4)
    train.add_argument("--batch-records", type=int, default=16)
    train.add_argument("--spatial-positions", type=int, default=512)

    evaluate = subcommands.add_parser("evaluate", help="state or actual no-replay evaluation")
    evaluate.add_argument("--artifact", required=True, type=Path)
    evaluate.add_argument("--pair-bank", action="append", type=Path)
    evaluate.add_argument("--pair-type", default="native", choices=("aligned", "native"))
    evaluate.add_argument("--split", default="dev", choices=("fit", "dev", "test"))
    evaluate.add_argument("--case-cache", type=Path)
    evaluate.add_argument("--sam2-repo", type=Path)
    evaluate.add_argument("--target-config")
    evaluate.add_argument("--target-checkpoint", type=Path)
    evaluate.add_argument("--target-model-id", default="sam2.1_hiera_base_plus")
    evaluate.add_argument("--video-dir", type=Path)
    evaluate.add_argument("--annotation-dir", type=Path)
    evaluate.add_argument("--artifact-dir", type=Path)
    evaluate.add_argument("--evaluation-repo", type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--seed", type=int, default=7)

    report = subcommands.add_parser("report", help="assemble report.md from completed stages")
    report.add_argument("--experiment-root", required=True, type=Path)
    return parser


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "download":
        _print(
            acquire_dataset(
                args.source,
                download_dir=args.download_dir,
                extract_dir=args.extract_dir,
                verify_checksum=not args.allow_checksum_mismatch,
            )
        )
        return
    if args.command == "audit":
        report = run_checkpoint_audit(
            sam2_repo=args.sam2_repo,
            source_config=args.source_config,
            source_checkpoint=args.source_checkpoint,
            target_config=args.target_config,
            target_checkpoint=args.target_checkpoint,
            output_dir=args.output_dir,
            device=args.device,
        )
        _print(report["inputs"])
        return
    if args.command == "split":
        manifest = build_train_split_manifest(
            args.dataset_root,
            dataset=args.dataset,
            release=args.release,
            official_split=args.official_split,
            seed=args.seed,
        )
        write_train_split_manifest(args.output, manifest)
        _print({"output": str(args.output.resolve()), "sha256": manifest["content_sha256"], "counts": manifest["counts"]})
        return
    if args.command == "extract":
        if args.full_train and args.tiny_overfit:
            raise ValueError("--full-train and --tiny-overfit are mutually exclusive")
        _print(
            extract_pair_bank(
                manifest_path=args.manifest,
                dataset_root=args.dataset_root,
                output_root=args.output_root,
                sam2_repo=args.sam2_repo,
                source_config=args.source_config,
                source_checkpoint=args.source_checkpoint,
                target_config=args.target_config,
                target_checkpoint=args.target_checkpoint,
                pair_types=tuple(args.pair_type or ("aligned", "native")),
                screening_only=not args.full_train,
                tiny_overfit_only=args.tiny_overfit,
                device=args.device,
            )
        )
        return
    if args.command == "fit-transforms":
        fit_manifest = json.loads(args.fit_manifest.read_text(encoding="utf-8"))
        payload = fit_transform_suite(
            load_banks(args.aligned_fit_bank),
            checkpoint_matrices=args.checkpoint_matrices,
            output=args.output,
            fit_manifest_sha256=fit_manifest["content_sha256"],
        )
        _print({"output": str(args.output.resolve()), "roundtrip": payload["roundtrip"]})
        return
    if args.command == "train":
        _print(
            train_two_stage(
                transform_suite=args.transform_suite,
                aligned_pairs=load_banks(args.aligned_bank),
                native_pairs=load_banks(args.native_bank),
                spatial_method=args.spatial_method,
                pointer_method=args.pointer_method,
                architecture=args.architecture,
                output_dir=args.output_dir,
                seed=args.seed,
                device=args.device,
                aligned_steps=args.aligned_steps,
                native_steps=args.native_steps,
                aligned_learning_rate=args.aligned_lr,
                native_learning_rate=args.native_lr,
                batch_records=args.batch_records,
                spatial_positions_per_record=args.spatial_positions,
            )
        )
        return
    if args.command == "evaluate":
        if bool(args.pair_bank) == bool(args.case_cache):
            raise ValueError("choose exactly one of --pair-bank or --case-cache")
        if args.pair_bank:
            _print(
                evaluate_translator_state(
                    args.artifact,
                    load_banks(args.pair_bank),
                    output=args.output,
                    pair_type=args.pair_type,
                    split=args.split,
                )
            )
            return
        required = {
            "sam2_repo": args.sam2_repo,
            "target_config": args.target_config,
            "target_checkpoint": args.target_checkpoint,
            "video_dir": args.video_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"no-replay evaluation is missing arguments: {missing}")
        payload = torch.load(args.artifact, map_location="cpu", weights_only=True)
        translator = PreconditionedStateTranslator.from_payload(payload).to(args.device).eval()
        report = run_cached_translator_handoff(
            case_cache=args.case_cache,
            sam2_repo=args.sam2_repo,
            target_config_file=args.target_config,
            target_checkpoint=args.target_checkpoint,
            target_model_id=args.target_model_id,
            video_dir=args.video_dir,
            annotation_dir=args.annotation_dir,
            device=args.device,
            seed=args.seed,
            translator=translator,
            translator_name="preconditioned",
            candidate_label="Preconditioned Translator",
            artifact_dir=(
                args.artifact_dir
                if args.artifact_dir is not None
                else (args.output.parent / args.output.stem)
                if args.annotation_dir is not None
                else None
            ),
        )
        prefix_calls = int(report["backbone_calls_before_injection"]) + int(report["backbone_calls_during_injection"])
        report["prefix_backbone_calls"] = prefix_calls
        report["no_replay_contract_satisfied"] = prefix_calls == 0
        if prefix_calls != 0:
            raise RuntimeError(f"no-replay contract violated: {prefix_calls} prefix backbone calls")
        if args.annotation_dir is not None:
            if args.evaluation_repo is None:
                raise ValueError("--evaluation-repo is required for GT J&F evaluation")
            artifact_dir = (
                args.artifact_dir
                if args.artifact_dir is not None
                else args.output.parent / args.output.stem
            ).resolve()
            frame_paths = sam2_frame_paths(args.video_dir)
            frame_id_map = {index: path.stem for index, path in enumerate(frame_paths)}
            iou_metric, boundary_metric, evaluator_commit = load_official_davis_metrics(
                args.evaluation_repo
            )
            metric_source = f"davisvideochallenge/davis2017-evaluation@{evaluator_commit}"
            common = {
                "annotation_directory": args.annotation_dir,
                "object_id": int(report["object_id"]),
                "start_frame": int(report["switch_frame"]) + 1,
                "iou_metric": iou_metric,
                "boundary_metric": boundary_metric,
                "metric_source": metric_source,
                "sequence": str(report["video_id"]),
                "frame_id_map": frame_id_map,
            }
            candidate = evaluate_davis_future_masks(
                prediction_directory=artifact_dir / "candidate_masks", **common
            )
            native = evaluate_davis_future_masks(
                prediction_directory=artifact_dir / "oracle_masks", **common
            )
            report["candidate_j_and_f"] = candidate
            report["target_native_j_and_f"] = native
            report["temporal"] = evaluate_temporal_handoff(
                candidate, native, switch_frame=int(report["switch_frame"])
            )
        atomic_write_text(args.output, json.dumps(report, indent=2, ensure_ascii=False))
        _print(report)
        return
    if args.command == "report":
        _print({"report": str(assemble_report(args.experiment_root))})
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
