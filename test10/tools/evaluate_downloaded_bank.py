#!/usr/bin/env python3
"""Frozen-checkpoint, first-five-frame GT evaluation of EVERY downloaded pair.

This is a new state-bank experiment, NOT a rescore of the old native-prefix run.
No training, prefix regeneration, target-oracle substitution, or test-time tuning.
Only future RGB backbone features are shared; each method owns its tracking state.
The downloaded target tensor is not called a native prediction reference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import WORKSPACE, REPO, digest, read, sha, snapshot, verify_snapshot, write

METHODS = ("direct", "affine", "residual_mlp", "transformer")
SCHEMA = "test10.downloaded_bank_first5.v1"
HORIZON = 5


def membership(report):
    train, validation = set(), set()
    for row in report["collection"]["pairs"]:
        (train if row["split"] == "train" else validation).add((row["dataset"], row["video_id"]))
    if train & validation:
        raise ValueError("training and checkpoint-selection video overlap")
    return train, validation


def role(dataset, video, train, validation):
    key = dataset, video
    return "training_video" if key in train else "checkpoint_selection_video" if key in validation else "excluded_from_training_and_selection"


def data_dirs(dataset, video):
    data = WORKSPACE / "vos-data"
    roots = ([data / "MOSEv2/train"] if dataset == "MOSEv2" else
             [data / "LVOS_V2" / split for split in ("train", "valid", "test")])
    matches = [r for r in roots if (r / "JPEGImages" / video).is_dir() and (r / "Annotations" / video).is_dir()]
    if len(matches) != 1:
        raise ValueError(f"ambiguous/missing RGB and annotations: {dataset}/{video}: {matches}")
    return matches[0] / "JPEGImages" / video, matches[0] / "Annotations" / video


def interleave(rows):
    groups = {d: sorted((r for r in rows if r["dataset"] == d), key=lambda r: (r["video_id"], r["pair_id"]))
              for d in ("MOSEv2", "LVOSv2")}
    return [group[i] for i in range(max(map(len, groups.values()))) for group in groups.values() if i < len(group)]


def audit(args):
    from run import ensure_local_training_module
    training = ensure_local_training_module()
    report = training.training_report(args.training_dir)
    train, validation = membership(report)
    catalog = read(args.catalog)
    if catalog.get("schema") != "test10.pair_selection.v1":
        raise ValueError("invalid downloaded pair catalog")
    metadata_models = {"source_model_id": "sam2.1-small", "target_model_id": "sam2.1-base-plus"}
    videos, hashes, cases, seen = {}, {}, [], set()
    for index, row in enumerate(catalog["pairs"], 1):
        path = Path(row["path"])
        meta_path = path.with_suffix(".prepare.json")
        meta = read(meta_path)
        expected = path.with_suffix(".pt.sha256").read_text().split()[0]
        if expected != row["sha256"] or meta["cache"]["sha256"] != expected:
            raise ValueError(f"inconsistent pair checksums: {path}")
        if meta["cache"]["bytes"] != path.stat().st_size:
            raise ValueError(f"pair size differs from sidecar: {path}")
        if any(meta.get(k) != v for k, v in metadata_models.items()) or meta["video_id"] != row["video_id"]:
            raise ValueError(f"pair source/target/video mismatch: {path}")
        pair_id = f"{row['dataset']}:{path.stem}"
        if pair_id in seen:
            raise ValueError("duplicate pair identifier")
        seen.add(pair_id)
        vkey = row["dataset"], row["video_id"]
        if vkey not in videos:
            rgb, annotations = data_dirs(*vkey)
            images = sorted((p for p in rgb.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")), key=lambda p: int(p.stem))
            stems = [p.stem for p in images]
            if row["dataset"] == "LVOSv2" and any(int(b)-int(a) != 5 for a,b in zip(stems, stems[1:])):
                raise ValueError("LVOS processed frame stride differs from the bank")
            videos[vkey] = dict(video_dir=str(rgb), annotation_dir=str(annotations), frame_stems=stems)
        inv = videos[vkey]
        switch = int(meta["switch_frame"])
        if meta["num_frames"] != len(inv["frame_stems"]) or switch < 0 or switch + HORIZON >= meta["num_frames"]:
            raise ValueError(f"missing/mismatched first-five frame timeline: {path}")
        inputs = {}
        for stem in inv["frame_stems"][switch+1:switch+HORIZON+1]:
            for directory, suffix in ((inv["video_dir"], ".jpg"), (inv["annotation_dir"], ".png")):
                f = Path(directory) / (stem + suffix)
                if suffix == ".jpg" and not f.is_file():
                    f = f.with_suffix(".jpeg")
                if not f.is_file():
                    raise ValueError(f"missing required RGB/GT: {f}")
                if str(f) not in hashes:
                    hashes[str(f)] = sha(f)
                inputs[str(f)] = hashes[str(f)]
        cases.append(dict(pair_id=pair_id, case_id=pair_id, dataset=row["dataset"], video_id=row["video_id"],
            object_id=int(meta["object_id"]), source_pool="fit" if row["split"] == "train" else "development",
            evaluation_role=role(*vkey, train, validation), pair_path=str(path), pair_sha256=expected,
            prepare_sha256=sha(meta_path), switch=switch, end=switch+HORIZON, input_hashes=inputs, **inv))
        if index % 2000 == 0:
            print(f"audit: {index}/{len(catalog['pairs'])} pairs", flush=True)
    from manifest import library_imports, imports
    library_imports()
    C = imports()
    paths = [args.catalog, Path(__file__), args.training_dir / "train_report.json", ROOT / "runtime.py",
             ROOT / "run.py", ROOT / "state_pairs.py", ROOT / "training.py", ROOT / "core.py", ROOT / "manifest.py"]
    paths += [args.training_dir / report["models"][m]["checkpoint"] for m in ("affine", "residual_mlp", "transformer")]
    paths += [C.MODELS["base_plus"]["checkpoint"]]
    paths += list((WORKSPACE / "sam2/sam2").rglob("*.py")) + list((WORKSPACE / "sam2/sam2/configs").rglob("*.yaml"))
    paths += list((REPO / "src").rglob("*.py")) + list((REPO / "scripts").glob("mvp_*.py"))
    counts = {d: dict(pairs=sum(c["dataset"] == d for c in cases),
                      videos=len({c["video_id"] for c in cases if c["dataset"] == d}),
                      by_role=dict(Counter(c["evaluation_role"] for c in cases if c["dataset"] == d)))
              for d in ("MOSEv2", "LVOSv2")}
    manifest = dict(schema=SCHEMA, horizon=HORIZON, methods=list(METHODS), cases=interleave(cases), counts=counts,
                    metric="mean J/F/J&F over processed switch+1..switch+5, then mean cases within video",
                    warning="Full bank includes training/selection videos. Disjoint-video summaries are separate. "
                            "Downloaded source states differ numerically from old local prefixes; do not merge runs. "
                            "Downloaded target is NOT a native full-prefix prediction reference.",
                    raw_strides={"MOSEv2": 1, "LVOSv2": 5})
    provenance = dict(schema=SCHEMA, files=snapshot(paths), catalog_sha256=sha(args.catalog),
        training_dir=str(args.training_dir.resolve()), checkpoints=report["models"], seed=7,
        source="downloaded_runpod_small_state", target="sam2.1-base-plus", autocast="bfloat16",
        translator_precision="fp32 outside autocast, source dtype restored", training=False,
        optimization="same-case future image backbone feature reuse; independent per-method memory and outputs",
        runtime_gate="local Base+ native vs reinjected full-logit equality on one smoke case per dataset; "
                     "all smoke methods additionally compare cached vs uncached full logits; "
                     "every case verifies exact output positions and zero past backbone calls",
        timing_scope="optimized throughput; NOT an independently measured handoff latency benchmark")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / "selection.json").exists():
        raise ValueError("audit refuses to overwrite an existing run")
    write(args.run_dir / "selection.json", manifest)
    write(args.run_dir / "provenance.json", provenance)
    write(args.run_dir / "audit.json", dict(counts=counts, total_pairs=len(cases),
        total_videos=len(videos), unique_future_input_files=len(hashes), eligible_pairs=len(cases)))
    print(json.dumps(counts, ensure_ascii=False, indent=2), flush=True)


class FutureFeatureCache:
    """Share only image-encoder outputs, never memory-conditioned features."""
    def __init__(self, predictor, positions):
        self.predictor, self.positions = predictor, set(positions)
        self.original = predictor._get_image_feature
        self.features, self.calls = {}, []

    def __enter__(self):
        def get(state, frame_idx, batch_size):
            if frame_idx not in self.positions:
                raise RuntimeError(f"past/unexpected feature request {frame_idx}")
            self.calls.append(frame_idx)
            if frame_idx in self.features:
                state["cached_features"][frame_idx] = self.features[frame_idx]
            result = self.original(state, frame_idx, batch_size)
            self.features[frame_idx] = state["cached_features"][frame_idx]
            return result
        self.predictor._get_image_feature = get
        return self

    def __exit__(self, *args):
        self.predictor._get_image_feature = self.original
        self.features.clear()


def continue_state(predictor, frames, canonical, case):
    import torch
    from runtime import Counter, fresh_inference_state, inference_context, propagate, logit_hash
    from vos_memory_inspector.sam2_state import inject_sam2_canonical_state
    positions = list(range(case["switch"]+1, case["end"]+1))
    with inference_context("bfloat16"), Counter(predictor) as counter:
        state = fresh_inference_state(predictor, frames)
        inject_sam2_canonical_state(canonical, predictor=predictor, inference_state=state)
        if counter.frames:
            raise RuntimeError("injection unexpectedly processed past RGB")
        masks, hashes = {}, {}
        for position, logits in propagate(predictor, state, start=positions[0], end=positions[-1]):
            hashes[position] = logit_hash(logits)
            masks[position] = (logits > 0).cpu().numpy()
        if list(masks) != positions or any(p not in positions for p in counter.frames):
            raise RuntimeError("wrong continuation output/backbone positions")
    return masks, hashes, dict(backbone_frames=counter.frames, past_backbone_calls=0)


def score_five(case, masks, annotations, voids):
    from vos_memory_inspector.vos_metrics import evaluate_case
    report = evaluate_case(masks, annotations, switch_frame=case["switch"], voids=voids)
    if report["post_switch"]["frames"] != HORIZON or report["post_switch"]["J_and_F"] is None:
        raise RuntimeError("first-five GT/F metric coverage is incomplete")
    return dict(average=report["post_switch"], frames=report["frames"],
                visible=report["gt_visible"], absent=report["gt_absent"], false_positives=report["false_positives"])


def local_native_gate(predictor, frames, source, case):
    from runtime import (fresh_inference_state, inference_context, load_label_mask,
                         propagate, logit_hash, active_state, canonicalize_sam2_inference_state)
    cond = source.frame_indices[source.validity & source.is_conditioning].tolist()
    if len(cond) != 1:
        raise ValueError("local native gate requires one conditioning anchor")
    first = cond[0]
    prompt = load_label_mask(Path(case["annotation_dir"]) / (case["frame_stems"][first]+".png"), case["object_id"])
    with inference_context("bfloat16"):
        native = fresh_inference_state(predictor, frames)
        predictor.add_new_mask(native, frame_idx=first, obj_id=1, mask=prompt)
        for _ in propagate(predictor, native, start=first, end=case["switch"]):
            pass
        canonical = active_state(canonicalize_sam2_inference_state(native, switch_frame=case["switch"], strict=True),
            num_maskmem=predictor.num_maskmem, max_obj_ptrs=predictor.max_obj_ptrs_in_encoder)
        hashes = {p: logit_hash(logits) for p, logits in propagate(predictor, native, start=case["switch"]+1, end=case["end"])}
    _, injected, _ = continue_state(predictor, frames, canonical, case)
    if hashes != injected:
        raise RuntimeError("local native self-injection gate failed")
    return dict(pair_id=case["pair_id"], dataset=case["dataset"], exact_full_logit_match=True,
                frames=HORIZON, note="Locally generated native-state gate; NOT downloaded target/native equivalence")


def row_path(run_dir, case, method):
    return run_dir / "scores" / case["dataset"] / (digest([case["pair_id"],method])+".json")


def valid_saved(path, case, method, fingerprint):
    if not path.exists():
        return False
    row = read(path)
    if (row.get("fingerprint") != fingerprint or row.get("pair_sha256") != case["pair_sha256"] or
        row.get("method") != method or row.get("pair_id") != case["pair_id"] or
        row.get("input_fingerprint") != digest(case["input_hashes"]) or row.get("horizon") != HORIZON or
        row.get("scores",{}).get("average",{}).get("frames") != HORIZON):
        raise ValueError(f"saved row contract mismatch: {path}")
    return True


def select_smoke(cases):
    selected = []
    for d in ("MOSEv2", "LVOSv2"):
        for pool in ("fit", "development"):
            seen = set()
            for c in cases:
                if c["dataset"] == d and c["source_pool"] == pool and c["video_id"] not in seen:
                    seen.add(c["video_id"]); selected.append(c)
                    if len(seen) == 2:
                        break
            if len(seen) != 2:
                raise ValueError("insufficient smoke videos")
    return selected


def perform(args):
    from run import ensure_local_training_module
    training = ensure_local_training_module()
    from manifest import imports
    from runtime import SharedVideoFrames, inference_context
    from state_pairs import load_pair
    from vos_memory_inspector.upstream import SUPPORTED_SAM2_COMMIT
    from vos_memory_inspector.vos_metrics import object_masks, read_indexed_png
    from vos_memory_inspector.translators import DirectCopyTranslator
    from vos_memory_inspector.transformer_translator import SAM21_MEMORY_SPEC
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    selection, prov = read(args.run_dir / "selection.json"), read(args.run_dir / "provenance.json")
    verify_snapshot(prov["files"])
    if selection["schema"] != SCHEMA or selection["methods"] != list(METHODS):
        raise ValueError("different experiment schema/methods")
    fingerprint = digest(prov)
    if args.stage == "full" and not read(args.run_dir / "smoke_gate.json")["passed"]:
        raise RuntimeError("smoke must pass before full")
    torch.manual_seed(prov["seed"])
    # Eager predictor, same bf16 computation as test10. No optimizer is created.
    C = imports()
    base = C.build_predictor("base_plus")
    models = training.load_models(Path(prov["training_dir"]), "cuda")
    models["direct"] = DirectCopyTranslator(SAM21_MEMORY_SPEC)
    device = dict(name=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda)
    if (args.run_dir / "device.json").exists() and read(args.run_dir / "device.json") != device:
        raise ValueError("GPU/runtime differs from the saved run")
    write(args.run_dir / "device.json", device)
    cases = select_smoke(selection["cases"]) if args.stage == "smoke" else selection["cases"]
    started, timings, checked, runtime_gates = time.monotonic(), [], [], []
    with ThreadPoolExecutor(max_workers=2) as scorers:
        for index, case in enumerate(cases, 1):
            # A completed row is reusable only while its RGB/GT still match.
            verify_snapshot(case["input_hashes"])
            if sha(Path(case["pair_path"]).with_suffix(".prepare.json")) != case["prepare_sha256"]:
                raise ValueError("pair prepare metadata changed")
            missing = [m for m in METHODS if not valid_saved(row_path(args.run_dir,case,m),case,m,fingerprint)]
            if not missing and args.stage == "full":
                continue
            t = time.monotonic()
            source, target, metadata = load_pair(case["pair_path"], expected_sha=case["pair_sha256"])
            if (metadata["video_id"] != case["video_id"] or metadata["object_id"] != case["object_id"] or
                metadata["switch_frame"] != case["switch"] or metadata["num_frames"] != len(case["frame_stems"]) or
                metadata["upstream_commit"] != SUPPORTED_SAM2_COMMIT or
                metadata["num_maskmem"] != base.num_maskmem or metadata["max_obj_ptrs_in_encoder"] != base.max_obj_ptrs_in_encoder):
                raise ValueError("loaded pair differs from model/evaluation contract")
            frames = SharedVideoFrames(case["video_dir"], device="cuda")
            try:
                if frames.stems != case["frame_stems"]:
                    raise ValueError("frame list changed after audit")
                positions = list(range(case["switch"]+1,case["end"]+1))
                frames.preload(positions)
                annotations, voids = {}, {}
                for p in positions:
                    annotations[p], voids[p] = object_masks(read_indexed_png(Path(case["annotation_dir"]) /
                        (case["frame_stems"][p]+".png")),case["object_id"])
                if args.stage == "smoke" and case["dataset"] not in {r["dataset"] for r in runtime_gates}:
                    runtime_gates.append(local_native_gate(base,frames,source,case))
                    frames.release(); frames.preload(positions)
                pending, baseline = {}, {}
                gpu_source = dataclasses.replace(source, spatial_memory=source.spatial_memory.cuda(),
                    object_pointer=source.object_pointer.cuda(), presence_logits=source.presence_logits.cuda())
                if args.stage == "smoke":
                    for method in METHODS:
                        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                            translated = models[method].translate(gpu_source)
                        _, baseline[method], counter = continue_state(base,frames,translated,case)
                        if counter["backbone_frames"] != positions:
                            raise RuntimeError("uncached smoke did not encode exactly five future frames")
                with FutureFeatureCache(base, positions) as shared:
                    for method in (METHODS if args.stage == "smoke" else missing):
                        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                            translated = models[method].translate(gpu_source)
                        mark = len(shared.calls)
                        masks, hashes, counter = continue_state(base,frames,translated,case)
                        if shared.calls[mark:] != positions:
                            raise RuntimeError("shared cache changed feature request order")
                        if args.stage == "smoke" and hashes != baseline[method]:
                            raise RuntimeError("shared backbone features changed full logits")
                        pending[method] = (scorers.submit(score_five,case,masks,annotations,voids),hashes,counter)
                    if set(shared.features) != set(positions):
                        raise RuntimeError("shared cache covers unexpected frames")
                for method, (future, hashes, counter) in pending.items():
                    path = row_path(args.run_dir,case,method)
                    scores = future.result()
                    hashes = {str(p): h for p,h in hashes.items()}
                    result = dict(schema=SCHEMA, fingerprint=fingerprint, pair_id=case["pair_id"],
                        dataset=case["dataset"], video_id=case["video_id"], object_id=case["object_id"],
                        source_pool=case["source_pool"], evaluation_role=case["evaluation_role"],
                        switch=case["switch"], frame_stems=[case["frame_stems"][p] for p in positions],
                        pair_sha256=case["pair_sha256"], input_fingerprint=digest(case["input_hashes"]),
                        method=method, horizon=HORIZON, logit_hashes=hashes, runtime=counter, scores=scores)
                    if path.exists():
                        old = read(path)
                        if old["logit_hashes"] != hashes or old["scores"] != scores:
                            raise RuntimeError("resumed smoke changed saved predictions/scores")
                    else:
                        write(path,result)
                checked.append(case["pair_id"])
            finally:
                frames.release()
            del gpu_source, source, target, translated
            elapsed = time.monotonic()-t
            timings.append(elapsed)
            if args.stage == "smoke" or index % 20 == 0 or index == len(cases):
                progress = dict(stage=args.stage, scheduled_pairs=len(selection["cases"]),
                    queue_index=index, queue_size=len(cases), session_completed_pairs=len(timings),
                    session_seconds=time.monotonic()-started, mean_pair_seconds=sum(timings)/len(timings),
                    last_pair_seconds=elapsed, dataset=case["dataset"], last_pair=case["pair_id"],
                    estimated_remaining_hours=(len(cases)-index)*(sum(timings)/len(timings))/3600)
                write(args.run_dir / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            if index % 1000 == 0:
                verify_snapshot(prov["files"])
    verify_snapshot(prov["files"])
    if args.stage == "smoke":
        write(args.run_dir / "smoke_gate.json", dict(passed=True,cases=checked, runtime_gates=runtime_gates,
            cached_vs_uncached_full_logits_equal=len(cases)*len(METHODS),
            methods=list(METHODS), native_gate_cases=len(runtime_gates), seconds=time.monotonic()-started,
            mean_smoke_pair_seconds=sum(timings)/len(timings), native_prefix_gate_overhead_included=True))
    else:
        summarize(args.run_dir)


def summarize(run_dir):
    """Complete four-method common cases only; pair and video weighting explicit."""
    import numpy as np
    selection, prov = read(run_dir / "selection.json"), read(run_dir / "provenance.json")
    fingerprint = digest(prov)
    rows, incomplete = [], []
    for c in selection["cases"]:
        absent = [m for m in METHODS if not valid_saved(row_path(run_dir,c,m),c,m,fingerprint)]
        if absent:
            incomplete.append(dict(pair_id=c["pair_id"], methods=absent)); continue
        rows.extend(read(row_path(run_dir,c,m)) for m in METHODS)
    def stats(a):
        a = np.asarray(a,dtype=np.float64)
        rng = np.random.default_rng(7)
        draws = np.concatenate([a[rng.integers(len(a),size=(min(100,2000-i),len(a)))].mean(1)
                                for i in range(0,2000,100)])
        return dict(mean=float(a.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),videos=len(a))
    groups = {}
    selectors = {"all": lambda r: True, "fit_bank": lambda r:r["source_pool"]=="fit",
                 "development_bank":lambda r:r["source_pool"]=="development",
                 "video_disjoint_from_current_training_and_selection":lambda r:r["evaluation_role"]=="excluded_from_training_and_selection",
                 "training_videos":lambda r:r["evaluation_role"]=="training_video",
                 "checkpoint_selection_videos":lambda r:r["evaluation_role"]=="checkpoint_selection_video"}
    for group, predicate in selectors.items():
        datasets = {}
        for d in ("MOSEv2","LVOSv2"):
            sub = [r for r in rows if r["dataset"]==d and predicate(r)]
            if not sub: continue
            video_values = defaultdict(lambda:defaultdict(list))
            for r in sub:
                video_values[r["method"]][r["video_id"]].append(r["scores"]["average"])
            table, paired = {}, {}
            for m in METHODS:
                values = video_values[m]
                table[m] = {field:stats([sum(v[field] for v in values[vid])/len(values[vid])*100
                             for vid in sorted(values)]) for field in ("J","F","J_and_F")}
            for a,b in (("affine","direct"),("residual_mlp","affine"),("transformer","affine"),("transformer","residual_mlp")):
                av,bv = video_values[a],video_values[b]
                if av.keys()!=bv.keys(): raise RuntimeError("paired video sets differ")
                paired[a+"-minus-"+b] = stats([(sum(x["J_and_F"] for x in av[v])/len(av[v])-
                    sum(x["J_and_F"] for x in bv[v])/len(bv[v]))*100 for v in sorted(av)])
            datasets[d] = dict(pairs=len(sub)//len(METHODS),videos=len(video_values["affine"]),methods=table,paired=paired)
        groups[group] = datasets
    result = dict(schema=SCHEMA,horizon=HORIZON,complete_pairs=len(rows)//len(METHODS),scheduled_pairs=len(selection["cases"]),
                  complete_method_rows=len(rows),incomplete=incomplete,groups=groups,
                  ci="video-clustered paired bootstrap, 2000 draws, NumPy seed 7; pointwise 95%, no multiplicity correction",
                  note=selection["warning"], checkpoints=prov["checkpoints"])
    write(run_dir / "summary.json",result)
    lines = ["# Downloaded full bank — frozen translators, first 5 frames", "", selection["warning"], "",
             f"Completed {result['complete_pairs']}/{result['scheduled_pairs']} pairs; {len(incomplete)} incomplete.", "",
             "No training. Scores 0–100; mean five frames per pair, mean pairs within video, equal video mean. "
             "CIs are pointwise, video-clustered bootstrap; multiple comparisons are not adjusted."]
    for group,datasets in groups.items():
        lines += ["", "## "+group,"", "| Dataset | Pairs | Videos | Direct | Affine | MLP | Transformer |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for d,tab in datasets.items():
            lines.append("| "+d+" | "+str(tab["pairs"])+" | "+str(tab["videos"])+" | "+" | ".join(
                f"{tab['methods'][m]['J_and_F']['mean']:.2f}" for m in METHODS)+" |")
        lines += ["", "Paired differences and 95% CI:", ""]
        for d,tab in datasets.items():
            for contrast,val in tab["paired"].items():
                lines.append(f"- {d} {contrast}: {val['mean']:+.2f} [{val['ci95'][0]:+.2f}, {val['ci95'][1]:+.2f}]")
    p=run_dir / "SUMMARY.md"
    temp=p.with_suffix(".partial");temp.write_text("\n".join(lines)+"\n",encoding="utf-8");temp.replace(p)
    print(json.dumps({"complete_pairs":result["complete_pairs"],"incomplete":len(incomplete)}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("audit","smoke","full","report"),required=True)
    p.add_argument("--catalog",type=Path,default=WORKSPACE/"test10_pair_selection.json")
    p.add_argument("--training-dir",type=Path,default=ROOT/"training/run_fit1000_val200")
    p.add_argument("--run-dir",type=Path,required=True)
    args=p.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=True)
    with (args.run_dir/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=="audit":audit(args)
        elif args.stage=="report":summarize(args.run_dir)
        else:perform(args)


if __name__=="__main__":
    main()
