"""Freeze a video-disjoint four-dataset evaluation before running any model."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse

import numpy as np
from PIL import Image

from core import WORKSPACE, ROOT, read, write, rank, sha
from manifest import make_case

DATASETS = ("MOSEv2", "LVOSv2", "DAVIS2017", "VOST")
METHODS = ("small_only", "base_native", "direct", "affine", "residual_mlp",
           "transformer", "last_mask", "anchor_replay_16")
PREVIOUS = ROOT / "runs/run01"
LVOS_MANIFEST = WORKSPACE / "test9/vos-memory-translator-nonlinear/manifests/lvosv2_valid_v1.json"


def frame_number(path):
    return int(path.stem.removeprefix("frame"))


def sorted_frames(directory, suffix):
    return sorted(Path(directory).glob(f"*.{suffix}"), key=frame_number)


def first_objects(annotation):
    labels = np.unique(np.asarray(Image.open(annotation)))
    return [int(x) for x in labels if int(x) not in (0, 255)]


def pick(rows, dataset, seed, count=35):
    """Select one video per slot, cycling across video-length thirds."""
    rows = sorted(rows, key=lambda r: (r["length"], r["video_id"]))
    bins = [[], [], []]
    for i, row in enumerate(rows):
        row = dict(row, length_bin=min(2, 3 * i // len(rows)))
        bins[row["length_bin"]].append(row)
    bins = [sorted(bucket, key=lambda r: rank(seed, dataset, r["video_id"])) for bucket in bins]
    selected = []
    while len(selected) < count and any(bins):
        for bucket in bins:
            if len(selected) < count and bucket:
                selected.append(bucket.pop(0))
    if len(selected) < count:
        raise ValueError(f"{dataset}: only {len(selected)} eligible videos")
    return selected


def link_vost(video, run_dir):
    source = WORKSPACE / "vos-data/VOST/VOST"
    target = run_dir / "views/VOST" / video
    for folder, suffix in (("JPEGImages", "jpg"), ("Annotations", "png")):
        source_dir = source / folder / video
        target_dir = target / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted_frames(source_dir, suffix):
            link = target_dir / f"{frame_number(path):05d}.{suffix}"
            if link.is_symlink():
                if link.resolve() != path.resolve():
                    raise ValueError(f"view link changed: {link}")
            elif link.exists():
                raise ValueError(f"view path is not a symlink: {link}")
            else:
                link.symlink_to(path.resolve())
    return target / "JPEGImages", target / "Annotations"


def candidates(seed):
    old = read(PREVIOUS / "selection.json")
    done = {a["case_id"] for a in read(PREVIOUS / "budget.json")["attempts"] if a["status"] == "complete"}
    mose = [dict(video_id=c["video_id"], length=c["end"] - c["first"] + 1, existing=c)
            for c in old["cases"] if c["dataset"] == "MOSEv2" and c["cohort"] == "additional"
            and c["case_id"] not in done]
    lvos_manifest = read(LVOS_MANIFEST)
    by_video = defaultdict(list)
    for item in lvos_manifest["cases"]:
        by_video[item["video_id"]].append(item)
    lvos = []
    root = WORKSPACE / "vos-data/LVOS_V2/valid/JPEGImages"
    for video, items in by_video.items():
        chosen = min(items, key=lambda x: rank(seed, "LVOSv2", video, x["object_id"]))
        frames = sorted_frames(root / video, "jpg")
        stems = [frame_number(p) for p in frames]
        first, end = int(chosen["first_prompt_frame"]), int(chosen["future_end_frame"])
        if first not in stems or end not in stems:
            raise ValueError(f"LVOS manifest/frame mismatch: {video}")
        lvos.append(dict(video_id=video, length=stems.index(end) - stems.index(first) + 1,
                         object_id=int(chosen["object_id"]), first=first, end=end))
    davis_root = WORKSPACE / "vos-data/DAVIS"
    davis = []
    for video in (davis_root / "ImageSets/2017/train.txt").read_text().split():
        frames = sorted_frames(davis_root / "JPEGImages/480p" / video, "jpg")
        labels = first_objects(davis_root / "Annotations/480p" / video / f"{frames[0].stem}.png")
        if not labels: raise ValueError(f"no DAVIS first-frame object: {video}")
        obj = min(labels, key=lambda x: rank(seed, "DAVIS2017", video, x))
        davis.append(dict(video_id=video, length=len(frames), object_id=obj,
                          first=frame_number(frames[0]), end=frame_number(frames[-1])))
    vost_root = WORKSPACE / "vos-data/VOST/VOST"
    vost = []
    for video in (vost_root / "ImageSets/val.txt").read_text().split():
        frames = sorted_frames(vost_root / "JPEGImages" / video, "jpg")
        annotations = sorted_frames(vost_root / "Annotations" / video, "png")
        if [frame_number(p) for p in frames] != [frame_number(p) for p in annotations]:
            raise ValueError(f"VOST RGB/annotation mismatch: {video}")
        labels = first_objects(annotations[0])
        if not labels: raise ValueError(f"no VOST first-frame object: {video}")
        obj = min(labels, key=lambda x: rank(seed, "VOST", video, x))
        vost.append(dict(video_id=video, length=len(frames), object_id=obj,
                         first=frame_number(frames[0]), end=frame_number(frames[-1])))
    return {"MOSEv2": mose, "LVOSv2": lvos, "DAVIS2017": davis, "VOST": vost}


def build(run_dir, seed=7):
    pools = candidates(seed)
    chosen = {d: pick(pools[d], d, seed) for d in DATASETS}
    cases = []
    for slot in range(35):
        for dataset in DATASETS:
            row = chosen[dataset][slot]
            if dataset == "MOSEv2":
                case = dict(row["existing"])
            else:
                video = row["video_id"]
                raw = dict(case_id=f"heldout:{dataset}:{video}:obj{row['object_id']}",
                           dataset=dataset, video_id=video, object_id=row["object_id"],
                           first_prompt_stem=row["first"], switch_stem=row["first"],
                           end_stem=row["end"], data_split="valid" if dataset == "LVOSv2" else
                           "val" if dataset == "VOST" else "train")
                if dataset == "VOST":
                    images, annotations = link_vost(video, run_dir)
                    raw.update(video_dir_override=str(images), annotation_dir_override=str(annotations))
                case = make_case(raw, set(), additional=True)
            case.update(cohort="heldout_core" if slot < 30 else "heldout_extension",
                        checkpoint_video=False, length_bin=row["length_bin"], slot=slot,
                        data_split="valid" if dataset == "LVOSv2" else "val" if dataset == "VOST" else "train")
            if dataset == "DAVIS2017": case["prior_selection_status"] = "exploratory_train_split"
            cases.append(case)
    fit = {d: set(v["fit"]) for d, v in read(PREVIOUS / "selection.json")["split_evidence"].items()
           if "fit" in v}
    for d in ("MOSEv2", "LVOSv2"):
        if fit[d] & {c["video_id"] for c in cases if c["dataset"] == d}:
            raise ValueError(f"fit/evaluation video overlap: {d}")
    if len({(c["dataset"], c["video_id"]) for c in cases}) != 140:
        raise ValueError("evaluation videos are not unique")
    return dict(schema="test10.v1", design="heldout.v1", seed=seed, methods=METHODS,
                cases=cases, source_hashes={str(p): sha(p) for p in (
                    PREVIOUS / "selection.json", PREVIOUS / "budget.json", LVOS_MANIFEST,
                    WORKSPACE / "vos-data/DAVIS/ImageSets/2017/train.txt",
                    WORKSPACE / "vos-data/VOST/VOST/ImageSets/val.txt")},
                candidate_counts={d: len(pools[d]) for d in DATASETS},
                note="30 videos per dataset primary; 5 per dataset predeclared extension; no score-based selection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    path = args.run_dir / "frozen_selection.json"
    if path.exists(): raise ValueError(f"selection already exists: {path}")
    write(path, build(args.run_dir.resolve(), args.seed))
    print(path)
