#!/usr/bin/env python3
"""Add temporal metrics to completed baseline suites without rerunning SAM 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vos_memory_inspector.baseline_sweep import (
    aggregate_completed,
    attach_temporal_metrics,
    case_slug,
    write_aggregate_reports,
    write_json_atomic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--suite-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    cases = selection["cases"]
    upgraded = 0
    for case in cases:
        suite = args.suite_root / case_slug(case)
        summary_path = suite / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = attach_temporal_metrics(summary, suite)
        write_json_atomic(summary, summary_path)
        upgraded += 1
    aggregate = aggregate_completed(cases, args.suite_root)
    write_aggregate_reports(aggregate, args.output_root)
    print(
        json.dumps(
            {
                "upgraded_suites": upgraded,
                "aggregate": str((args.output_root / "aggregate.json").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
