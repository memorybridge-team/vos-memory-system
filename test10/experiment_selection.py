"""Freeze the planned 1,000 train + 200 checkpoint-selection fit pairs."""
from collections import defaultdict
from pathlib import Path

from core import rank, read, write


def build(catalog, seed=7):
    rows = read(catalog)["pairs"]
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            grouped[row["dataset"], row["video_id"]].append(row)
    selected = []
    counts = {}
    for dataset in ("MOSEv2", "LVOSv2"):
        videos = sorted((v for d, v in grouped if d == dataset),
                        key=lambda v: rank(seed, dataset, v))
        if dataset == "MOSEv2":
            n_train_videos, n_val_videos, n_train, n_val = 500, 100, 500, 100
        else:
            n_train_videos, n_val_videos, n_train, n_val = 250, 75, 500, 100
        if len(videos) < n_train_videos + n_val_videos:
            raise ValueError(f"not enough fit videos for {dataset}")
        partitions = {"train": videos[:n_train_videos],
                      "validation": videos[n_train_videos:n_train_videos+n_val_videos]}
        for split, target in (("train", n_train), ("validation", n_val)):
            pool = []
            for video in partitions[split]:
                candidates = sorted(grouped[dataset, video],
                                    key=lambda row: rank(seed, dataset, video, row["path"]))
                pool.extend(candidates)
            pool.sort(key=lambda row: rank(seed, dataset, split, row["video_id"], row["path"]))
            # Round-robin per video limits concentration in object-rich videos.
            by_video = defaultdict(list)
            for row in pool:
                by_video[row["video_id"]].append(row)
            chosen = []
            depth = 0
            ordered_videos = sorted(partitions[split], key=lambda v: rank(seed, dataset, split, v))
            while len(chosen) < target:
                added = False
                for video in ordered_videos:
                    if depth < len(by_video[video]):
                        chosen.append(by_video[video][depth]); added = True
                        if len(chosen) == target:
                            break
                if not added:
                    raise ValueError(f"only {len(chosen)} fit pairs available for {dataset}/{split}")
                depth += 1
            selected.extend(dict(row, split=split) for row in chosen)
            counts[f"{dataset}/{split}"] = dict(pairs=len(chosen), videos=len({r["video_id"] for r in chosen}),
                                                  max_pairs_per_video=max(sum(r["video_id"] == v for r in chosen)
                                                                          for v in {r["video_id"] for r in chosen}))
    return dict(schema="test10.pair_selection.v1", source="fit_only_frozen_2026-10-01",
                seed=seed, counts=counts, pairs=selected)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    result = build(args.catalog, 7)
    write(args.output, result)
    print(result["counts"])
