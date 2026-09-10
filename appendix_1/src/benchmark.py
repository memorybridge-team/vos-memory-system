#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from .runner import experiment_plan, run_benchmark
from .summarize import import_server_results, summarize_run


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "benchmark.json"


def main():
    parser = argparse.ArgumentParser(
        description="SAM2 cold versus native prefilled-memory benchmark")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-videos", type=int,
                        help="run only the first N videos per dataset (marked partial)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="print matrix without downloading")
    commands.add_parser("run", help="download on demand and run/resume")
    summary = commands.add_parser("summarize", help="aggregate a run")
    summary.add_argument("--run-id")
    imported = commands.add_parser("import-server-results")
    imported.add_argument("--file", required=True)
    imported.add_argument("--submission-id", required=True)
    imported.add_argument("--model", required=True)
    imported.add_argument("--condition", choices=("cold", "full_memory"), required=True)
    imported.add_argument("--run-id")
    args = parser.parse_args()
    if args.command == "plan":
        result = experiment_plan(args.config, args.max_videos)
    elif args.command == "run":
        result = run_benchmark(args.config, args.max_videos)
    elif args.command == "summarize":
        result = summarize_run(args.config, args.run_id)
    else:
        result = import_server_results(
            args.config, args.file, args.submission_id, args.model,
            args.condition, args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
