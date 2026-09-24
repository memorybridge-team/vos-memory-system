from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compatibility import compare_manifests, write_compatibility_report
from .davis import download_davis_2017_trainval_480p, validate_davis_sequence
from .davis_evaluation import (
    evaluate_davis_future_masks,
    load_official_davis_metrics,
    write_davis_future_report,
)
from .evaluation_manifest import (
    build_davis_evaluation_manifest,
    write_evaluation_manifest,
)
from .mose import build_mosev2_evaluation_manifest, build_mosev2_train_manifest
from .lvos import build_lvosv2_evaluation_manifest
from .paired_experiment import (
    load_case_cache_pair,
    load_canonical_state,
    run_paired_experiment,
    run_synthetic_experiment,
)
from .runner import run_video_probe
from .roundtrip import (
    MaskPromptEvent,
    prepare_cross_model_case_reference,
    run_cached_baseline,
    run_cached_translator_handoff,
    run_cross_model_direct_handoff,
    run_cross_model_translator_handoff,
    run_same_checkpoint_correction_roundtrip,
    run_same_checkpoint_prompt_timeline_roundtrip,
    run_same_checkpoint_repeated_switch_roundtrip,
    run_same_checkpoint_roundtrip,
)
from .state_inspector import inspect_state, write_inspection_report
from .translators import (
    ResidualMLPStateTranslator,
    RidgeDirectPresenceTranslator,
    RidgeStateTranslator,
)


def _probe_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe stored and consumed SAM 2 temporal-memory tensors."
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument(
        "--dump-tensor",
        action="append",
        default=[],
        help=(
            "Opt-in tensor name to save on CPU; repeat for multiple names. "
            "Without this option only statistics are written."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--allow-upstream-mismatch", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--canonical-state",
        type=Path,
        help="Opt-in .pt export of the continuation-oriented canonical state.",
    )
    return parser


def probe_main(argv: list[str] | None = None) -> None:
    args = _probe_parser().parse_args(argv)
    summary = run_video_probe(
        sam2_repo=args.sam2_repo,
        config_file=args.config,
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        object_id=args.object_id,
        switch_frame=args.switch_frame,
        jsonl_path=args.jsonl,
        csv_path=args.csv,
        dump_dir=args.dump_dir,
        dump_tensors=tuple(args.dump_tensor),
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        allow_upstream_mismatch=args.allow_upstream_mismatch,
        seed=args.seed,
        canonical_state_path=args.canonical_state,
    )
    print(json.dumps(summary, indent=2))


def compare_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare matching rows from sequential source and target manifests."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    report = compare_manifests(args.source, args.target)
    write_compatibility_report(report, args.json, args.markdown)
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, indent=2))


def davis_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate a DAVIS 2017 sequence and resolve probe inputs."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--resolution", default="480p")
    parser.add_argument(
        "--split",
        choices=("train", "val", "none"),
        default="val",
        help="Dataset split membership to validate. Use 'none' to skip membership checking.",
    )
    args = parser.parse_args(argv)
    sequence = validate_davis_sequence(
        args.root,
        args.sequence,
        resolution=args.resolution,
        split=None if args.split == "none" else args.split,
    )
    print(
        json.dumps(
            {
                "name": sequence.name,
                "frames_directory": str(sequence.frames_directory),
                "first_mask": str(sequence.first_mask),
                "frame_count": sequence.frame_count,
                "resolution": sequence.resolution,
            },
            indent=2,
        )
    )


def davis_download_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download and safely extract official DAVIS 2017 trainval 480p."
    )
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--accept-dataset-terms", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args(argv)
    root = download_davis_2017_trainval_480p(
        args.destination,
        accept_dataset_terms=args.accept_dataset_terms,
        keep_archive=args.keep_archive,
    )
    print(json.dumps({"davis_root": str(root)}, indent=2))


def davis_manifest_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic DAVIS video/object/switch manifest."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--resolution", default="480p")
    parser.add_argument(
        "--regular-quantile",
        action="append",
        type=float,
        default=[],
        help="Repeatable regular switch quantile. Default: 0.25, 0.5, 0.75.",
    )
    parser.add_argument("--min-prefix-frames", type=int, default=5)
    parser.add_argument("--min-future-frames", type=int, default=20)
    parser.add_argument("--area-drop-ratio", type=float, default=0.35)
    parser.add_argument("--area-growth-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_davis_evaluation_manifest(
        args.root,
        split=args.split,
        resolution=args.resolution,
        regular_quantiles=tuple(args.regular_quantile) or (0.25, 0.5, 0.75),
        min_prefix_frames=args.min_prefix_frames,
        min_future_frames=args.min_future_frames,
        area_drop_ratio=args.area_drop_ratio,
        area_growth_ratio=args.area_growth_ratio,
        seed=args.seed,
    )
    write_evaluation_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "split": manifest["split"],
                "sequences": manifest["sequence_count"],
                "cases": manifest["case_count"],
                "excluded": len(manifest["excluded"]),
                "content_sha256": manifest["content_sha256"],
            },
            indent=2,
        )
    )


def davis_future_evaluation_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate switch-future masks with official DAVIS J/F functions."
    )
    parser.add_argument("--evaluation-repo", required=True, type=Path)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--annotation-dir", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--object-id", required=True, type=int)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument(
        "--end-frame",
        type=int,
        help=(
            "Inclusive last frame. DAVIS semi-supervised evaluation excludes "
            "the video last frame."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    iou_metric, boundary_metric, commit = load_official_davis_metrics(
        args.evaluation_repo
    )
    report = evaluate_davis_future_masks(
        prediction_directory=args.prediction_dir,
        annotation_directory=args.annotation_dir,
        object_id=args.object_id,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        iou_metric=iou_metric,
        boundary_metric=boundary_metric,
        metric_source=f"davisvideochallenge/davis2017-evaluation@{commit}",
        sequence=args.sequence,
    )
    write_davis_future_report(report, args.output)
    print(json.dumps(report, indent=2))


def mose_manifest_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic MOSEv2 validation manifest. "
            "Future GT is marked unavailable because validation publishes only the first mask."
        )
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", default="valid")
    parser.add_argument(
        "--regular-quantile",
        action="append",
        type=float,
        default=[],
        help="Repeatable switch quantile. Default: 0.25, 0.5, 0.75.",
    )
    parser.add_argument("--min-prefix-frames", type=int, default=5)
    parser.add_argument("--min-future-frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    builder = build_mosev2_train_manifest if args.split == "train" else build_mosev2_evaluation_manifest
    manifest = builder(
        args.root, split=args.split,
        regular_quantiles=tuple(args.regular_quantile) or (0.25, 0.5, 0.75),
        min_prefix_frames=args.min_prefix_frames,
        min_future_frames=args.min_future_frames, seed=args.seed,
    )
    write_evaluation_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "split": manifest["split"],
                "sequences": manifest["sequence_count"],
                "cases": manifest["case_count"],
                "excluded": len(manifest["excluded"]),
                "content_sha256": manifest["content_sha256"],
            },
            indent=2,
        )
    )


def lvos_manifest_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic LVOS v2 metadata-driven evaluation manifest."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--regular-quantile", action="append", type=float, default=[])
    parser.add_argument("--min-prefix-frames", type=int, default=5)
    parser.add_argument("--min-future-frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_lvosv2_evaluation_manifest(
        args.root,
        split=args.split,
        regular_quantiles=tuple(args.regular_quantile) or (0.25, 0.5, 0.75),
        min_prefix_frames=args.min_prefix_frames,
        min_future_frames=args.min_future_frames,
        seed=args.seed,
    )
    write_evaluation_manifest(manifest, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "split": manifest["split"],
        "sequences": manifest["sequence_count"],
        "cases": manifest["case_count"],
        "excluded": len(manifest["excluded"]),
        "content_sha256": manifest["content_sha256"],
    }, indent=2))


def state_inspect_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Inspect tensors in a nested .pt state, HF cache, or canonical state."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--key", help="Optional top-level mapping key to inspect")
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    # State files are pickle-backed. Only load files from a trusted source.
    value = __import__("torch").load(args.input, map_location="cpu", weights_only=False)
    if args.key is not None:
        value = value[args.key]
    report = inspect_state(value)
    write_inspection_report(report, args.json, args.markdown)
    print(json.dumps(report.to_dict(), indent=2))


def synthetic_experiment_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run an explicitly synthetic paired-state translator smoke test."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--ridge-lambda", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=32)
    args = parser.parse_args(argv)
    report = run_synthetic_experiment(
        args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        ridge_lambda=args.ridge_lambda,
        hidden_dim=args.hidden_dim,
    )
    print(json.dumps(report, indent=2))


def paired_experiment_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit translators from paired canonical states and evaluate held-out pairs."
    )
    parser.add_argument("--train-source", action="append", type=Path, default=[])
    parser.add_argument("--train-target", action="append", type=Path, default=[])
    parser.add_argument("--test-source", action="append", type=Path, default=[])
    parser.add_argument("--test-target", action="append", type=Path, default=[])
    parser.add_argument("--train-case-cache", action="append", type=Path, default=[])
    parser.add_argument("--test-case-cache", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--ridge-lambda", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument(
        "--spatial-samples-per-pair",
        type=int,
        help="Sample this many valid spatial tokens per state and epoch.",
    )
    parser.add_argument(
        "--translator",
        action="append",
        choices=("direct", "ridge", "linear", "residual_mlp"),
        default=[],
        help="Translator to run; repeat as needed. Default: all.",
    )
    args = parser.parse_args(argv)
    if len(args.train_source) != len(args.train_target):
        parser.error("provide the same number of --train-source/--train-target")
    if len(args.test_source) != len(args.test_target):
        parser.error("provide the same number of --test-source/--test-target")
    train_pairs = [
        (load_canonical_state(source), load_canonical_state(target))
        for source, target in zip(args.train_source, args.train_target, strict=True)
    ]
    train_pairs.extend(load_case_cache_pair(path) for path in args.train_case_cache)
    test_pairs = [
        (load_canonical_state(source), load_canonical_state(target))
        for source, target in zip(args.test_source, args.test_target, strict=True)
    ]
    test_pairs.extend(load_case_cache_pair(path) for path in args.test_case_cache)
    if not train_pairs or not test_pairs:
        parser.error("provide at least one train pair and one test pair")
    report = run_paired_experiment(
        train_pairs,
        test_pairs,
        args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        ridge_lambda=args.ridge_lambda,
        hidden_dim=args.hidden_dim,
        translator_names=tuple(args.translator) or None,
        device=args.device,
        spatial_samples_per_pair=args.spatial_samples_per_pair,
    )
    print(json.dumps(report, indent=2))


def roundtrip_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run same-checkpoint SAM 2 export/inject continuation validation."
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    report = run_same_checkpoint_roundtrip(
        sam2_repo=args.sam2_repo,
        config_file=args.config,
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        object_id=args.object_id,
        switch_frame=args.switch_frame,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def prompt_timeline_roundtrip_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run same-checkpoint SAM 2 export/inject validation for a multi-object "
            "mask-prompt timeline."
        )
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt-event",
        action="append",
        nargs=3,
        metavar=("FRAME", "OBJECT_ID", "MASK_PATH"),
        required=True,
        help="Repeat for each user mask prompt in the timeline.",
    )
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    events = [
        MaskPromptEvent(
            frame_index=int(frame),
            object_id=int(object_id),
            mask_path=Path(mask_path),
        )
        for frame, object_id, mask_path in args.prompt_event
    ]
    report = run_same_checkpoint_prompt_timeline_roundtrip(
        sam2_repo=args.sam2_repo,
        config_file=args.config,
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        video_dir=args.video_dir,
        prompt_events=events,
        switch_frame=args.switch_frame,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def correction_roundtrip_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a same-checkpoint correction after handoff or the safe "
            "Target replay fallback for a correction before handoff."
        )
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt-event",
        action="append",
        nargs=3,
        metavar=("FRAME", "OBJECT_ID", "MASK_PATH"),
        required=True,
    )
    parser.add_argument(
        "--correction-event",
        nargs=3,
        metavar=("FRAME", "OBJECT_ID", "MASK_PATH"),
        required=True,
    )
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    events = [
        MaskPromptEvent(int(frame), int(object_id), Path(mask_path))
        for frame, object_id, mask_path in args.prompt_event
    ]
    correction = MaskPromptEvent(
        int(args.correction_event[0]),
        int(args.correction_event[1]),
        Path(args.correction_event[2]),
    )
    report = run_same_checkpoint_correction_roundtrip(
        sam2_repo=args.sam2_repo,
        config_file=args.config,
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        video_dir=args.video_dir,
        prompt_events=events,
        correction_event=correction,
        switch_frame=args.switch_frame,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def repeated_switch_roundtrip_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate two consecutive same-checkpoint SAM 2 handoffs."
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt-event",
        action="append",
        nargs=3,
        metavar=("FRAME", "OBJECT_ID", "MASK_PATH"),
        required=True,
    )
    parser.add_argument(
        "--switch-frames",
        nargs=2,
        type=int,
        metavar=("FIRST", "SECOND"),
        required=True,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    events = [
        MaskPromptEvent(int(frame), int(object_id), Path(mask_path))
        for frame, object_id, mask_path in args.prompt_event
    ]
    report = run_same_checkpoint_repeated_switch_roundtrip(
        sam2_repo=args.sam2_repo,
        config_file=args.config,
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        video_dir=args.video_dir,
        prompt_events=events,
        switch_frames=(args.switch_frames[0], args.switch_frames[1]),
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def prepare_handoff_case_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute one source prefix and target oracle once, then cache them for "
            "reuse by all handoff baselines."
        )
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--object-id", required=True, type=int)
    parser.add_argument("--switch-frame", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    report = prepare_cross_model_case_reference(
        sam2_repo=args.sam2_repo,
        source_config_file=args.source_config,
        source_checkpoint=args.source_checkpoint,
        source_model_id=args.source_model_id,
        target_config_file=args.target_config,
        target_checkpoint=args.target_checkpoint,
        target_model_id=args.target_model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        object_id=args.object_id,
        switch_frame=args.switch_frame,
        output=args.output,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def cached_handoff_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Direct, Ridge, or residual-MLP target continuation from a "
            "prepared case cache."
        )
    )
    parser.add_argument("--case-cache", required=True, type=Path)
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--annotation-dir", type=Path)
    parser.add_argument(
        "--translator",
        choices=("direct", "ridge", "residual_mlp"),
        default="direct",
    )
    parser.add_argument("--translator-artifact", type=Path)
    parser.add_argument(
        "--presence-policy", choices=("direct", "ridge"), default="direct"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)

    translator = None
    translator_name = "direct_copy"
    candidate_label = "Direct Copy"
    if args.translator == "ridge":
        if args.translator_artifact is None:
            parser.error("--translator-artifact is required for Ridge")
        payload = __import__("torch").load(
            args.translator_artifact, map_location="cpu", weights_only=True
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("ridge"), dict):
            parser.error("translator artifact does not contain a Ridge payload")
        ridge = RidgeStateTranslator.from_payload(payload["ridge"])
        if args.presence_policy == "direct":
            translator = RidgeDirectPresenceTranslator(ridge)
            translator_name = translator.name
            candidate_label = "Ridge memory/pointer + Direct presence"
        else:
            translator = ridge
            translator_name = ridge.name
            candidate_label = "Ridge"
    elif args.translator == "residual_mlp":
        if args.translator_artifact is None:
            parser.error("--translator-artifact is required for residual_mlp")
        payload = __import__("torch").load(
            args.translator_artifact, map_location="cpu", weights_only=True
        )
        mlp_payload = payload.get("residual_mlp") if isinstance(payload, dict) else None
        if not isinstance(mlp_payload, dict):
            parser.error("translator artifact does not contain a residual_mlp payload")
        translator = ResidualMLPStateTranslator.from_payload(mlp_payload)
        translator_name = translator.name
        candidate_label = "Nonlinear Residual MLP"
    report = run_cached_translator_handoff(
        case_cache=args.case_cache,
        sam2_repo=args.sam2_repo,
        target_config_file=args.target_config,
        target_checkpoint=args.target_checkpoint,
        target_model_id=args.target_model_id,
        video_dir=args.video_dir,
        annotation_dir=args.annotation_dir,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
        translator=translator,
        translator_name=translator_name,
        candidate_label=candidate_label,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def cached_baseline_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Target Reset, Last-Mask, Replay-k, or Full Replay against a "
            "prepared SAM 2 case cache."
        )
    )
    parser.add_argument(
        "--baseline",
        required=True,
        choices=("target_reset", "last_mask", "replay_k", "full_replay"),
    )
    parser.add_argument("--case-cache", required=True, type=Path)
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt-mask",
        type=Path,
        help="First-frame ground-truth mask; required only for full_replay.",
    )
    parser.add_argument("--annotation-dir", type=Path)
    parser.add_argument(
        "--replay-frames",
        type=int,
        help=(
            "Number of target-processed prefix frames including the switch frame. "
            "Required for replay_k; replay-1 intentionally equals Last-Mask."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)
    if args.baseline == "replay_k" and args.replay_frames is None:
        parser.error("--replay-frames is required for replay_k")
    if args.baseline == "full_replay" and args.prompt_mask is None:
        parser.error("--prompt-mask is required for full_replay")
    report = run_cached_baseline(
        baseline=args.baseline,
        case_cache=args.case_cache,
        sam2_repo=args.sam2_repo,
        target_config_file=args.target_config,
        target_checkpoint=args.target_checkpoint,
        target_model_id=args.target_model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        annotation_dir=args.annotation_dir,
        replay_frames=args.replay_frames,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def direct_handoff_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a checkpoint-backed cross-model Direct Copy handoff."
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Write browsable mask PNGs plus report.json/report.md.",
    )
    args = parser.parse_args(argv)
    report = run_cross_model_direct_handoff(
        sam2_repo=args.sam2_repo,
        source_config_file=args.source_config,
        source_checkpoint=args.source_checkpoint,
        source_model_id=args.source_model_id,
        target_config_file=args.target_config,
        target_checkpoint=args.target_checkpoint,
        target_model_id=args.target_model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        object_id=args.object_id,
        switch_frame=args.switch_frame,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def ridge_handoff_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run checkpoint-backed cross-model handoff using a saved Ridge model."
        )
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--translator-artifact", required=True, type=Path)
    parser.add_argument(
        "--presence-policy",
        choices=("direct", "ridge"),
        default="direct",
        help="Use source presence logits directly or apply the fitted Ridge head.",
    )
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto uses CUDA, then Apple MPS when allowed, otherwise CPU.",
    )
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)

    payload = __import__("torch").load(
        args.translator_artifact, map_location="cpu", weights_only=True
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("ridge"), dict):
        parser.error("translator artifact does not contain a Ridge payload")
    ridge = RidgeStateTranslator.from_payload(payload["ridge"])
    if args.presence_policy == "direct":
        translator = RidgeDirectPresenceTranslator(ridge)
        translator_name = translator.name
        candidate_label = "Ridge memory/pointer + Direct presence"
    else:
        translator = ridge
        translator_name = ridge.name
        candidate_label = "Ridge"
    report = run_cross_model_translator_handoff(
        sam2_repo=args.sam2_repo,
        source_config_file=args.source_config,
        source_checkpoint=args.source_checkpoint,
        source_model_id=args.source_model_id,
        target_config_file=args.target_config,
        target_checkpoint=args.target_checkpoint,
        target_model_id=args.target_model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        object_id=args.object_id,
        switch_frame=args.switch_frame,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
        translator=translator,
        translator_name=translator_name,
        candidate_label=candidate_label,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
