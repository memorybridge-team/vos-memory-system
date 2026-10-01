#!/usr/bin/env python3
"""Build a test10 training selection from downloaded RunPod state-only pairs."""
import argparse
from pathlib import Path

from core import read, write
from state_pairs import MODELS, SCHEMA

SPLITS = {"fit": "train", "development": "validation"}
DATASETS = {"MOSEv2": "mosev2", "LVOSv2": "lvosv2"}


def build(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"pair root does not exist: {root}")
    rows = []
    total = {}
    for dataset, manifest_prefix in DATASETS.items():
        split_videos = {}
        for source_split, training_split in SPLITS.items():
            manifest = read(root / "manifests" / f"{manifest_prefix}_train_v1_{source_split}.json")
            if (manifest.get("schema_version") != "cmmt.video_split_manifest.v1" or
                    manifest.get("split") != source_split or
                    manifest.get("dataset") != ("LVOS v2" if dataset == "LVOSv2" else dataset)):
                raise ValueError(f"split manifest mismatch: {dataset}/{source_split}")
            videos = set(manifest["videos"])
            if len(videos) != manifest["video_count"]:
                raise ValueError(f"duplicate or missing split videos: {dataset}/{source_split}")
            split_videos[source_split] = videos
            directory = root / dataset / source_split
            pair_files = sorted(directory.glob("*.pt"))
            if len(pair_files) != manifest["case_count"]:
                raise ValueError(f"incomplete pair directory: {dataset}/{source_split}: "
                                 f"{len(pair_files)}/{manifest['case_count']}")
            observed_videos = set()
            for path in pair_files:
                sidecar = read(path.with_suffix(".prepare.json"))
                checksum_file = path.with_suffix(".pt.sha256")
                parts = checksum_file.read_text(encoding="ascii").split()
                if (len(parts) != 2 or len(parts[0]) != 64 or parts[1] != path.name or
                        sidecar.get("cache", {}).get("sha256") != parts[0] or
                        sidecar["cache"].get("schema_version") != SCHEMA or
                        sidecar["cache"].get("bytes") != path.stat().st_size):
                    raise ValueError(f"pair sidecar mismatch: {path}")
                if (any(sidecar.get(k) != v for k, v in MODELS.items()) or
                        sidecar.get("cache_mode") != "state_only" or
                        sidecar.get("active_memory_only") is not True):
                    raise ValueError(f"not a Small→Base+ state-only pair: {path}")
                video = sidecar["video_id"]
                if video not in videos:
                    raise ValueError(f"pair video outside split: {path}")
                observed_videos.add(video)
                rows.append(dict(dataset=dataset, video_id=video, split=training_split,
                                 path=str(path), sha256=parts[0]))
            if observed_videos != videos:
                raise ValueError(f"split videos missing pairs: {dataset}/{source_split}")
            total[f"{dataset}/{source_split}"] = len(pair_files)
        if split_videos["fit"] & split_videos["development"]:
            raise ValueError(f"fit/development video overlap: {dataset}")
    return dict(schema="test10.pair_selection.v1", source="downloaded_runpod_state_pairs",
                counts=total, pairs=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="directory containing MOSEv2, LVOSv2 and manifests")
    parser.add_argument("--output", type=Path, required=True, help="new test10.pair_selection.v1 JSON")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    selection = build(args.root)
    write(args.output, selection)
    print(f"wrote {len(selection['pairs'])} pairs to {args.output}")
    print(selection["counts"])


if __name__ == "__main__":
    main()
