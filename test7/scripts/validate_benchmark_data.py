"""Validate that benchmark manifests resolve to real RGB/prompt-mask inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def validate(dataset: str, root: Path, manifest: dict, split_dir: str | None) -> dict:
    checked_prompts: set[tuple[str, int, int]] = set()
    failures: list[str] = []
    frame_cache: dict[str, list[Path]] = {}
    mask_cache: dict[str, list[Path]] = {}
    label_cache: dict[Path, set[int]] = {}
    for case in manifest["cases"]:
        video = str(case.get("video_id", case.get("sequence")))
        object_id = int(case["object_id"])
        if dataset == "davis":
            frames = frame_cache.setdefault(video, sorted((root / "JPEGImages" / "480p" / video).glob("*.jpg")))
            masks = mask_cache.setdefault(video, sorted((root / "Annotations" / "480p" / video).glob("*.png")))
            prompt_index = 0
        else:
            base = root / split_dir if split_dir and (root / split_dir).is_dir() else root
            frames = frame_cache.setdefault(video, sorted((base / "JPEGImages" / video).glob("*.jpg")))
            masks = mask_cache.setdefault(video, sorted((base / "Annotations" / video).glob("*.png")))
            if dataset == "mose":
                prompt_index = int(case.get("first_prompt_frame", 0))
            else:
                prompt_frame = int(case.get("first_prompt_frame", 0))
                mask_by_id = {int(path.stem): path for path in masks}
                prompt_index = next((i for i, path in enumerate(frames) if int(path.stem) == prompt_frame), -1)
                if prompt_frame not in mask_by_id:
                    failures.append(f"{video}: missing LVOS prompt mask {prompt_frame}")
                    continue
                masks = [mask_by_id[prompt_frame]]
                prompt_index = 0
        if not frames or not masks:
            failures.append(f"{video}: missing frames or masks")
            continue
        switch = int(case["switch_frame"])
        if dataset == "lvos":
            frame_ids = {int(path.stem) for path in frames}
            if switch not in frame_ids:
                failures.append(f"{video}: switch frame {switch} is absent")
        elif not 0 <= switch < len(frames):
            failures.append(f"{video}: switch index {switch} outside {len(frames)} frames")
        key = (video, object_id, int(case.get("first_prompt_frame", 0)))
        if key not in checked_prompts:
            checked_prompts.add(key)
            mask_path = masks[prompt_index] if dataset != "lvos" else masks[0]
            if mask_path not in label_cache:
                with Image.open(mask_path) as image:
                    label_cache[mask_path] = {int(value) for value in np.unique(np.asarray(image))}
            if object_id not in label_cache[mask_path]:
                failures.append(f"{video}: object {object_id} absent from {mask_path.name}")
    return {
        "dataset": dataset,
        "manifest_cases": len(manifest["cases"]),
        "unique_prompt_inputs": len(checked_prompts),
        "failure_count": len(failures),
        "failures": failures[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("davis", "mose", "lvos"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-dir")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = validate(args.dataset, args.root, manifest, args.split_dir)
    print(json.dumps(report, indent=2))
    if report["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
