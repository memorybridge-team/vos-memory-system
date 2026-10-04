"""Read existing splits and build deterministic, content-addressed candidates."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import sys

from core import (WORKSPACE, REPO, LEGACY, ROOT, SWITCH_WINDOW, DATASETS, PROTOCOL,
                  digest, read, sha, rank, positions, snapshot)


def library_imports():
    """Training needs translator classes, without test9/SAM2 runtime imports."""
    for path in (REPO / "scripts", REPO / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def imports():
    library_imports()
    import mvp_common
    return mvp_common


def data_paths(dataset, video, split="train"):
    data = WORKSPACE / "vos-data"
    roots = {"MOSEv2": data / "MOSEv2/train", "LVOSv2": data / "LVOS_V2" / split,
             "DAVIS2017": data / "DAVIS"}
    resolution = Path("480p") if dataset == "DAVIS2017" else Path()
    return roots[dataset] / "JPEGImages" / resolution / video, roots[dataset] / "Annotations" / resolution / video


@lru_cache(maxsize=None)
def inventory(dataset, video, split="train", image_override=None, annotation_override=None):
    if image_override and annotation_override:
        image_dir, ann_dir = Path(image_override), Path(annotation_override)
    else:
        image_dir, ann_dir = data_paths(dataset, video, split)
    images = sorted((p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")), key=lambda p: int(p.stem))
    if not images:
        raise ValueError("no images")
    stems = [p.stem for p in images]
    # LVOS data already contains every fifth raw frame. Never subsample it twice.
    if dataset == "LVOSv2" and any(int(b) - int(a) != 5 for a, b in zip(stems, stems[1:])):
        raise ValueError("LVOS sampling differs from test9 (expected raw stride 5)")
    annotations = {p.stem: sha(p) for p in sorted(ann_dir.glob("*.png"))}
    return dict(frame_stems=stems, image_files=[str(p) for p in images],
                image_hashes=[sha(p) for p in images], annotations=annotations,
                video_dir=str(image_dir), annotation_dir=str(ann_dir))


def normalize(raw, dataset):
    return dict(case_id=raw["case_id"], dataset=dataset, video_id=raw.get("video_id", raw.get("sequence")),
                object_id=int(raw["object_id"]), first_prompt_stem=int(raw.get("first_prompt_frame", raw.get("first_prompt_stem", 0))),
                switch_stem=int(raw.get("switch_frame", raw.get("switch_stem"))),
                end_stem=int(raw.get("future_end_frame", raw.get("end_stem"))))


def make_case(raw, checkpoint_videos, *, additional=False, preserve_switch=False):
    d, v = raw["dataset"], raw["video_id"]
    inv = inventory(d, v, raw.get("data_split", "train"), raw.get("video_dir_override"),
                    raw.get("annotation_dir_override")); stems = inv["frame_stems"]
    first = [int(s) for s in stems].index(raw["first_prompt_stem"])
    end = [int(s) for s in stems].index(raw["end_stem"])
    proposed = stems[max(first + 16, (first + end) // 2)] if additional and first + 16 < end else raw["switch_stem"]
    if preserve_switch or raw.get("state_pair_path"):
        switch = [int(s) for s in stems].index(raw["switch_stem"])
        if not first <= switch < end:
            raise ValueError("invalid prepared-pair switch position")
    else:
        first, switch, end = positions(stems, raw["first_prompt_stem"], proposed, raw["end_stem"])
    if end - switch < SWITCH_WINDOW:
        raise ValueError("fewer than ten post-switch processed frames")
    if stems[first] not in inv["annotations"]:
        raise ValueError("missing prompt annotation")
    if not any(s in inv["annotations"] for s in stems[switch + 1:end + 1]):
        raise ValueError("no annotated suffix")
    changed = int(stems[switch]) != raw["switch_stem"]
    cid = f"{d}:{v}:obj{raw['object_id']}:switch{stems[switch]}"
    return dict(raw, case_id=cid, original_case_id=raw["case_id"], changed=changed,
                cohort="additional" if additional else "legacy_dev",
                checkpoint_video=(d, v) in checkpoint_videos,
                # Prior DAVIS pilot use cannot be ruled out from the available metadata.
                prior_selection_status="unknown_exploratory" if d == "DAVIS2017" else "known",
                first=first, switch=switch, end=end, switch_stem=int(stems[switch]),
                frame_stems=stems, sampling={"raw_stride": 5 if d == "LVOSv2" else 6 if d == "VOST" else 1},
                input_sha256=digest(list(zip(stems[first:end + 1], inv["image_hashes"][first:end + 1]))),
                annotation_sha256=digest({s: inv["annotations"].get(s) for s in [stems[first], *stems[switch:end + 1]]}),
                video_dir=inv["video_dir"], annotation_dir=inv["annotation_dir"])


def build(seed=7):
    C = imports(); old = read(LEGACY / "selection.json")
    ckids = set(read(LEGACY / "ckpt_cases.json")["case_ids"])
    ckvids = {(c["dataset"], c["video_id"]) for c in old["dev_cases"] if c["case_id"] in ckids}
    oldvids = {(c["dataset"], c["video_id"]) for c in old["dev_cases"]}
    cases, rejected, pools = [], [], {}
    split_evidence = {}
    for d, names in C.MANIFESTS.items():
        source, fit, dev = [C.historical_manifest(n) for n in names]
        fitvids, devvids = set(fit["videos"]), set(dev["videos"])
        if fitvids & devvids:
            raise ValueError(f"fit/dev overlap: {d}")
        split_evidence[d] = {"fit": sorted(fitvids), "development": sorted(devvids), "source_digest": digest(source)}
        pools[d] = [normalize(c, d) for c in source["cases"] if c["video_id"] in devvids]
    # DAVIS validation objects from the already-versioned manifest; no future masks used for switch selection.
    davis = C.historical_manifest("davis2017_val_v1")
    val = set((WORKSPACE / "vos-data/DAVIS/ImageSets/2017/val.txt").read_text().split())
    pools["DAVIS2017"] = [normalize(c, "DAVIS2017") for c in davis["cases"] if c.get("video_id", c.get("sequence")) in val]
    split_evidence["DAVIS2017"] = {"validation": sorted(val), "source_digest": digest(davis)}
    used = set(); counts = defaultdict(int)
    def append(raw, additional):
        try:
            c = make_case(raw, ckvids, additional=additional)
            if c["case_id"] in used:
                return False
            cases.append(c); used.add(c["case_id"]); counts[(c["dataset"], c["video_id"])] += 1
            return True
        except (ValueError, FileNotFoundError, KeyError, IndexError) as exc:
            rejected.append({"case_id": raw["case_id"], "reason": str(exc)})
            return False
    for raw in old["dev_cases"]:
        append(raw, False)
    for d, target, cap in (("MOSEv2", 300, 1), ("LVOSv2", 100, 2)):
        missing = target - sum(c["dataset"] == d for c in cases)
        for raw in sorted(pools[d], key=lambda c: rank(seed, d, c["case_id"])):
            if missing <= 0: break
            if counts[(d, raw["video_id"])] < cap and append(raw, False): missing -= 1
        if missing:
            raise ValueError(f"cannot maintain legacy cohort: {d} missing {missing}")
    usedvids = {(c["dataset"], c["video_id"]) for c in cases} | oldvids | ckvids
    for d, pool in pools.items():
        for raw in sorted(pool, key=lambda c: rank(seed, d, c["video_id"], c["object_id"])):
            video = (d, raw["video_id"])
            if video not in usedvids and append(raw, True):
                usedvids.add(video)
    for d in DATASETS:
        group = sorted((c for c in cases if c["dataset"] == d), key=lambda c: (c["end"] - c["first"], c["case_id"]))
        for i, c in enumerate(group): c["length_bin"] = min(2, 3 * i // len(group))
    return dict(schema=PROTOCOL, seed=seed, cases=cases, rejected=rejected, split_evidence=split_evidence)


def verify_case(case):
    # Fresh content checks before a case; cached inventory is for manifest generation only.
    inventory.cache_clear()
    raw = dict(case, switch_stem=int(case["frame_stems"][case["switch"]]))
    current = make_case(raw, set(), additional=False, preserve_switch=True)
    for field in ("input_sha256", "annotation_sha256", "frame_stems"):
        if current[field] != case[field]: raise ValueError(f"case inputs changed: {case['case_id']} {field}")


def provenance(training_dir, *, include_legacy=True):
    C = imports()
    import torch
    import numpy
    import PIL
    paths = list((REPO / "src").rglob("*.py")) + list((REPO / "scripts").glob("mvp_*.py"))
    paths += list(ROOT.glob("*.py"))
    from training import training_report
    trained = training_report(training_dir)
    paths += [Path(training_dir) / "train_report.json"]
    paths += [Path(training_dir) / row["checkpoint"] for row in trained["models"].values()]
    legacy_paths = []
    if include_legacy:
        candidates = [LEGACY / "selection.json", LEGACY / "ckpt_cases.json", LEGACY / "eval_ckpt/selection.json",
                      LEGACY / "eval_dev/results.jsonl", *(LEGACY / "eval_dev").glob("run_meta_*.json")]
        legacy_paths = [p for p in candidates if p.is_file()]
        paths += legacy_paths
    paths += list((WORKSPACE / "sam2/sam2").rglob("*.py")) + list((WORKSPACE / "sam2/sam2/configs").rglob("*.yaml"))
    paths += [C.MODELS[k]["checkpoint"] for k in C.MODELS]
    return dict(protocol=PROTOCOL, files=snapshot(paths), models=C.model_provenance(),
                legacy_source_digest=C.code_revision()["source_digest"], autocast="bfloat16",
                training=trained, training_dir=str(Path(training_dir).resolve()),
                legacy=dict(requested=include_legacy,
                            results_available=include_legacy and (LEGACY / "eval_dev/results.jsonl").is_file(),
                            files=[str(p.resolve()) for p in legacy_paths]),
                state_pair_schema="cmmt.prepared_handoff_case.v2", switch_window=10,
                software={"python": sys.version, "torch": torch.__version__,
                                        "numpy": numpy.__version__, "PIL": PIL.__version__})


def audit_legacy(selection, prov):
    """Never promote unverifiable historical scores into the main comparison."""
    cases = [c for c in selection["cases"] if c.get("cohort") == "legacy_dev"]
    if not cases:
        return []
    results_path = LEGACY / "eval_dev/results.jsonl"
    available = results_path.is_file()
    metas = [read(p) for p in (LEGACY / "eval_dev").glob("run_meta_*.json")] if available else []
    matching = [m for m in metas if m["code_revision"]["source_digest"] == prov["legacy_source_digest"]
                and m["models"] == prov["models"] and m.get("autocast") == prov["autocast"]]
    rows = [__import__("json").loads(line) for line in results_path.read_text().splitlines()] if available else []
    index = {(r["dataset"], r["case_id"], r["method"]): r for r in rows}
    output = []
    for c in cases:
        for method in ("small_only", "direct", "linear", "residual_mlp", "base", "moment_match"):
            row = index.get((c["dataset"], c["original_case_id"], method))
            if not available: reason = "missing legacy results file"
            elif row is None: reason = "missing result"
            elif c["changed"]: reason = "switch changed"
            elif not matching: reason = "code/model/autocast mismatch"
            elif row["switch_position"] != c["switch"] or row["future_positions"] != c["end"] - c["switch"]:
                reason = "frame interval mismatch"
            else:
                reason = "historical RGB/annotation hashes and translator checkpoint hashes were not recorded"
            output.append(dict(case_id=c["case_id"], method=method, status="unverified" if row and not c["changed"] else "unusable",
                               reason=reason, historical_row=row))
    return output
