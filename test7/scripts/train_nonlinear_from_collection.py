"""Train the nonlinear translator from a completed paired-state collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vos_memory_inspector.paired_experiment import run_paired_experiment
from vos_memory_inspector.paired_state_cache import load_paired_state_cache


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load the video-level split from a paired-state selection manifest "
            "and train only the nonlinear residual MLP."
        )
    )
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--spatial-samples-per-pair", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _case_slug(case: dict[str, Any]) -> str:
    return (
        f"{case['sequence']}_obj{int(case['object_id'])}_"
        f"switch{int(case['switch_frame'])}"
    )


def _split_paths(
    selection: dict[str, Any], cache_root: Path
) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {"train": [], "validation": []}
    for case in selection["cases"]:
        split = str(case.get("paired_split"))
        if split not in result:
            raise ValueError(f"case has unsupported paired_split: {split!r}")
        path = cache_root / split / f"{_case_slug(case)}.pt"
        checksum = path.with_suffix(path.suffix + ".sha256")
        if not path.is_file() or not checksum.is_file():
            raise FileNotFoundError(f"paired case is incomplete: {path}")
        result[split].append(path)
    if not result["train"] or not result["validation"]:
        raise ValueError("selection needs non-empty train and validation cases")
    return result


def main() -> None:
    args = _parser().parse_args()
    selection_path = args.selection_manifest.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "cmmt.paired_state_selection.v1":
        raise ValueError("unsupported paired-state selection manifest")
    paths = _split_paths(selection, args.cache_root.resolve())
    train_pairs = [load_paired_state_cache(path)[:2] for path in paths["train"]]
    validation_pairs = [
        load_paired_state_cache(path)[:2] for path in paths["validation"]
    ]
    report = run_paired_experiment(
        train_pairs,
        validation_pairs,
        args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        translator_names=("residual_mlp",),
        device=args.device,
        spatial_samples_per_pair=args.spatial_samples_per_pair,
    )
    report["input_collection"] = {
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": _sha256(selection_path),
        "train": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in paths["train"]
        ],
        "validation": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in paths["validation"]
        ],
    }
    report_path = args.output_dir.resolve() / "paired_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
