#!/usr/bin/env python3
"""Run a small official-VOST Base+ identity-handoff diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from vos_memory_inspector.vost_roundtrip import (
    VIDEO_IDS,
    load_clip,
    run_experiment,
    validate_downloads,
)


def main() -> int:
    project = PROJECT
    workspace = project.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project / "data/vost_subset/val")
    parser.add_argument("--sam2-repo", type=Path, default=workspace / "sam2")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=workspace / "sam2/checkpoints/sam2.1_hiera_base_plus.pt",
    )
    parser.add_argument(
        "--evaluator-repo", type=Path, default=project / ".external/vost-evaluation"
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check-data-only", action="store_true")
    args = parser.parse_args()

    if args.check_data_only:
        downloads = validate_downloads(args.data_root)
        clips = [load_clip(args.data_root, video_id) for video_id in VIDEO_IDS]
        print(json.dumps({
            "status": "data_validated",
            "downloads": downloads,
            "clips": [{
                "video_id": clip.video_id,
                "frames": len(clip.frames),
                "original_frame_range": [clip.original_frame_ids[0], clip.original_frame_ids[-1]],
                "switch_index": clip.switch_index,
                "switch_original_frame_id": clip.original_frame_ids[clip.switch_index],
                "object_ids": [prompt.object_id for prompt in clip.prompts],
            } for clip in clips],
        }, indent=2))
        return 0

    output_root = args.output_root or (
        project / "outputs/vost_base_roundtrip" /
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    report = run_experiment(
        data_root=args.data_root,
        sam2_repo=args.sam2_repo,
        checkpoint=args.checkpoint,
        evaluator_repo=args.evaluator_repo,
        output_root=output_root,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps({
        "status": report["status"],
        "report": str(output_root / "report.json"),
    }, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
