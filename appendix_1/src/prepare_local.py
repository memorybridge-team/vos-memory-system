#!/usr/bin/env python3
"""Download datasets locally and create optional transfer/storage shards."""

import argparse
import json
import shutil
from pathlib import Path

from .datasets import acquire, discover


DATASETS = (
    {"name": "davis2017", "split": "val"},
    {"name": "lvos", "version": "v1", "split": "val"},
    {"name": "mosev2", "split": "train"},
    {"name": "mosev2", "split": "valid"},
)


def _safe_video_name(name):
    path = Path(name)
    if path.name != name or name in ("", ".", ".."):
        raise ValueError(f"Unsafe video name: {name!r}")
    return name


def _copy_video(source, destination, video_name):
    source = Path(source) / video_name
    destination = Path(destination) / video_name
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _metadata_path(root, split):
    paths = list(Path(root).rglob(f"meta_{split}.json"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one meta_{split}.json under {root}")
    return paths[0]


def _image_root(root, spec):
    """Find the single extracted JPEGImages root, including nested archives."""
    candidates = list(Path(root).rglob("JPEGImages"))
    if spec["name"] == "davis2017":
        candidates = [path for path in candidates if (path / "480p").is_dir()]
    else:
        aliases = {spec["split"], "valid" if spec["split"] == "val" else spec["split"]}
        preferred = [path for path in candidates if path.parent.name in aliases]
        if preferred:
            candidates = preferred
    if len(candidates) != 1:
        raise ValueError(f"Expected one staging JPEGImages root under {root}, found {candidates}")
    return candidates[0]


def stage_dataset(spec, source_root, output_root, video_names):
    """Copy only selected videos while preserving the adapter's expected layout."""
    source_root, output_root = Path(source_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected = [_safe_video_name(name) for name in video_names]
    selected_set = set(selected)
    if len(selected_set) != len(selected):
        raise ValueError("Duplicate video names in shard")

    source_videos = {video.name: video for video in discover(spec, source_root)}
    missing = sorted(selected_set - set(source_videos))
    if missing:
        raise ValueError(f"Videos are not present in source split: {missing}")

    name, split = spec["name"], spec["split"]
    if name == "davis2017":
        image_root = _image_root(source_root, spec) / "480p"
        dataset_root = image_root.parent.parent
        mask_root = dataset_root / "Annotations" / "480p"
        out_image_root = output_root / "JPEGImages" / "480p"
        out_mask_root = output_root / "Annotations" / "480p"
        for video in selected:
            _copy_video(image_root, out_image_root, video)
            _copy_video(mask_root, out_mask_root, video)
        split_file = dataset_root / "ImageSets" / "2017" / f"{split}.txt"
        out_split = output_root / "ImageSets" / "2017" / f"{split}.txt"
        out_split.parent.mkdir(parents=True, exist_ok=True)
        out_split.write_text("\n".join(selected) + "\n")
    elif name == "lvos":
        image_root = _image_root(source_root, spec)
        mask_root = image_root.parent / "Annotations"
        out_image_root = output_root / "JPEGImages"
        out_mask_root = output_root / "Annotations"
        for video in selected:
            _copy_video(image_root, out_image_root, video)
            _copy_video(mask_root, out_mask_root, video)
        metadata = next(
            path for path in source_root.rglob("*meta.json")
            if path.name in ("val_meta.json", "valid_meta.json")
        )
        payload = json.loads(metadata.read_text())
        payload["videos"] = {video: payload["videos"][video] for video in selected}
        metadata_out = output_root / "metadata" / metadata.name
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.write_text(json.dumps(payload, indent=2) + "\n")
    elif name == "mosev2":
        image_root = _image_root(source_root, spec)
        mask_root = image_root.parent / "Annotations"
        out_image_root = output_root / "JPEGImages"
        out_mask_root = output_root / "Annotations"
        for video in selected:
            _copy_video(image_root, out_image_root, video)
            _copy_video(mask_root, out_mask_root, video)
        payload = json.loads(_metadata_path(source_root, split).read_text())
        payload["videos"] = {video: payload["videos"][video] for video in selected}
        (output_root / f"meta_{split}.json").write_text(json.dumps(payload, indent=2) + "\n")
    else:
        raise ValueError(f"Unsupported dataset: {name}/{split}")

    manifest = {
        "dataset": name,
        "split": split,
        "source_root": str(source_root.resolve()),
        "videos": selected,
        "video_count": len(selected),
    }
    (output_root / ".stage_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Validate the staged tree before it is transferred.
    discover(spec, output_root)
    return manifest


def download_all(cache):
    cache = Path(cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in DATASETS:
        print(f"[download] {spec['name']}/{spec['split']}", flush=True)
        root = acquire(spec, cache)
        videos = discover(spec, root)
        results.append({"dataset": spec["name"], "split": spec["split"],
                        "root": str(root), "videos": len(videos)})
        print(f"[ready] {root} ({len(videos)} videos)", flush=True)
    manifest = cache / "local-datasets.json"
    manifest.write_text(json.dumps({"datasets": results}, indent=2) + "\n")
    print(manifest)
    return results


def _default_cache():
    return Path(__file__).resolve().parents[1] / "cache"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=_default_cache(),
                        help="local cache containing downloaded and extracted datasets")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download-all", help="download and validate all four requested splits")
    stage = commands.add_parser("stage", help="create a size-limited dataset shard")
    stage.add_argument("--dataset", choices=("davis2017", "lvos", "mosev2"), required=True)
    stage.add_argument("--split", choices=("val", "train", "valid"), required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--videos", nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "download-all":
        download_all(args.cache)
        return
    spec = {"name": args.dataset, "split": args.split}
    if args.dataset == "lvos":
        spec["version"] = "v1"
    source = args.cache / "datasets" / args.dataset / args.split
    result = stage_dataset(spec, source, args.output, args.videos)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
