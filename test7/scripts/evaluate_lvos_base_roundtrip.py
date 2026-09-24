#!/usr/bin/env python3
"""Run or preflight the fourteen-clip LVOS Base+ identity handoff diagnostic."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from vos_memory_inspector.base_roundtrip import (
    VIDEO_IDS,
    load_clip,
    run_experiment,
    validate_downloads,
)


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    workspace = project.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project / "data/lvosv2_subset/val")
    parser.add_argument("--manifest", type=Path, default=project / "manifests/lvosv2_valid_v1.json")
    parser.add_argument("--sam2-repo", type=Path, default=workspace / "sam2")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=workspace / "sam2/checkpoints/sam2.1_hiera_base_plus.pt",
    )
    parser.add_argument("--metrics-repo", type=Path, default=project / ".external/lvos-evaluation")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="auto", help="auto uses CUDA if available, otherwise CPU")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check-data-only", action="store_true")
    args = parser.parse_args()

    if args.check_data_only:
        selection = json.loads(args.manifest.read_text(encoding="utf-8"))
        clips = [load_clip(args.data_root, selection, video_id) for video_id in VIDEO_IDS]
        print(json.dumps({
            "status": "data_validated",
            "downloads": validate_downloads(args.data_root),
            "clips": [{
                "video_id": clip.video_id,
                "frames": len(clip.frames),
                "switch_original_frame_id": clip.original_frame_ids[clip.switch_index],
                "prompts": [prompt.__dict__ for prompt in clip.prompts],
            } for clip in clips],
        }, indent=2))
        return 0

    output_root = args.output_root or (
        project / "outputs/lvos_base_roundtrip" /
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    report = run_experiment(
        data_root=args.data_root,
        manifest_path=args.manifest,
        sam2_repo=args.sam2_repo,
        checkpoint=args.checkpoint,
        metrics_repo=args.metrics_repo,
        output_root=output_root,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "report": str(output_root / "report.json")}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
