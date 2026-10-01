"""GPU evaluation adapters. Imported only for an explicit smoke/full invocation."""
from __future__ import annotations

import dataclasses
import hashlib
import os
import shutil
import time
from pathlib import Path

from core import EPOCHS, LEGACY, METHODS, GATE, sha, write, read, key, suffix_scores
from core import rescore_case as core_rescore_case
from manifest import imports, verify_case

C = imports()
import torch
from mvp_scoring import pack, score
from vos_memory_inspector.sam2_session import (SharedVideoFrames, fresh_inference_state,
    inference_context, load_label_mask, propagate)
from vos_memory_inspector.sam2_state import canonicalize_sam2_inference_state, inject_sam2_canonical_state
from vos_memory_inspector.transformer_translator import build_translator, SAM21_MEMORY_SPEC
from vos_memory_inspector.translators import DirectCopyTranslator, LearnedComponentPolicyTranslator


def now():
    torch.cuda.synchronize()
    return time.perf_counter()


def logit_hash(mask):
    # Compare complete float32 values, not only thresholded masks; storage is O(frames).
    cpu = mask.detach().float().cpu().contiguous()
    if not torch.isfinite(cpu).all():
        raise ValueError("non-finite prediction")
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


class Counter:
    """Record actual backbone frame indices, including hidden warmup/prefix calls."""
    def __init__(self, predictor):
        self.p = predictor
        self.forward, self.feature = predictor.forward_image, predictor._get_image_feature
        self.current = None
        self.frames = []

    def __enter__(self):
        def feature(inference_state, frame_idx, batch_size):
            self.current = int(frame_idx)
            return self.feature(inference_state, frame_idx, batch_size)
        def forward(image):
            self.frames.append(self.current)
            return self.forward(image)
        self.p._get_image_feature, self.p.forward_image = feature, forward
        return self

    def __exit__(self, *args):
        self.p.forward_image, self.p._get_image_feature = self.forward, self.feature


def save_blob(store, case, name, payload):
    path = store.path(case, name, ".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    torch.save(payload, temp)
    os.replace(temp, path)
    write(path.with_suffix(".sha.json"), {"sha256": sha(path), "key": key(case, name, store.provenance)})


def load_blob(store, case, name):
    path = store.path(case, name, ".pt")
    marker = path.with_suffix(".sha.json")
    if not path.exists() or not marker.exists():
        for root in store.reuse_roots:
            candidate = root / "artifacts" / path.name
            if candidate.is_file() and candidate.with_suffix(".sha.json").is_file():
                path, marker = candidate, candidate.with_suffix(".sha.json")
                break
        else:
            return None
    info = read(marker)
    if info["key"] != key(case, name, store.provenance) or sha(path) != info["sha256"]:
        raise ValueError(f"corrupt cache: {path}")
    # Only self-created, content-verified local artifacts contain CanonicalState objects.
    return torch.load(path, map_location="cpu", weights_only=False)


def score_positions(case, method, masks):
    """test9 scorer on any contiguous post-switch prefix given by the mask positions."""
    stems = case["frame_stems"]
    result = score(dict(case_id=case["case_id"], method=method, masks=masks,
                        stems={p: stems[p] for p in masks}, annotation_dir=case["annotation_dir"],
                        object_id=case["object_id"], switch_position=case["switch"],
                        context_position=case["switch"], stems_context=stems[case["switch"]]))
    return result["scores"]


def score_masks(case, method, masks):
    return suffix_scores(case, masks, lambda subset: score_positions(case, method, subset))


def rescore_case(case, store, methods):
    """CPU-only: rescore existing rows from sha-verified prediction caches."""
    cache = {}

    def blob(name):
        if name not in cache:
            cache[name] = load_blob(store, case, name)
            if cache[name] is None:
                raise FileNotFoundError(f"missing cache {name} for {case['case_id']}")
        return cache[name]

    def masks_for(method):
        name = {"small_only": "source_prefix", "base_native": "base_prefix"}.get(method, method + "_predictions")
        return blob(name)["masks"]

    return core_rescore_case(case, store, methods, masks_for, score_masks)


class Evaluator:
    def __init__(self, seed=7):
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for smoke/full")
        torch.manual_seed(seed)
        t0 = now()
        self.small, self.base = C.build_predictor("small"), C.build_predictor("base_plus")
        self.model_load_s = now() - t0
        self.methods = {"direct": DirectCopyTranslator(SAM21_MEMORY_SPEC)}
        for preset, epoch in EPOCHS.items():
            payload = torch.load(LEGACY / "train" / preset / f"epoch_{epoch:03d}.pt", map_location="cpu", weights_only=True)
            model = build_translator(preset, SAM21_MEMORY_SPEC, SAM21_MEMORY_SPEC).cuda().eval()
            model.load_state_dict(payload["state_dict"])
            name = {"linear": "affine", "base": "transformer"}.get(preset, preset)
            self.methods[name] = model
        affine = self.methods["affine"]
        self.methods["affine_spatial"] = LearnedComponentPolicyTranslator(affine, learned_components=("spatial_memory",))
        self.methods["affine_pointer"] = LearnedComponentPolicyTranslator(affine, learned_components=("object_pointer",))

    def prefix(self, case, frames, model, store, name):
        cached = load_blob(store, case, name)
        if cached is not None: return cached
        state = fresh_inference_state(model, frames)
        prompt = load_label_mask(Path(case["annotation_dir"]) / f"{case['frame_stems'][case['first']]}.png", case["object_id"])
        with inference_context("bfloat16"), Counter(model) as counter:
            started = now()
            model.add_new_mask(state, frame_idx=case["first"], obj_id=1, mask=prompt)
            seen = []
            for pos, mask in propagate(model, state, start=case["first"], end=case["switch"]):
                seen.append(pos)
            if seen != list(range(case["first"], case["switch"] + 1)):
                raise ValueError("prefix frame mismatch")
            prefix_s = now() - started
            last_mask = (mask > 0).cpu().numpy()
            started = now()
            canonical = canonicalize_sam2_inference_state(state, switch_frame=case["switch"], strict=True)
            export_s = now() - started
            # Preserve the true native suffix, not a suffix produced by reinjection.
            masks, hashes = {}, {}
            cont_start = now()
            for pos, mask in propagate(model, state, start=case["switch"] + 1, end=case["end"]):
                masks[pos] = pack((mask > 0).cpu().numpy())
                if name == "base_prefix": hashes[pos] = logit_hash(mask)
            continuation_s = now() - cont_start
            if counter.frames != list(range(case["first"], case["end"] + 1)):
                raise RuntimeError("native prefix/suffix backbone calls differ from frame timeline")
        payload = dict(state=canonical, last_mask=last_mask, masks=masks, logit_hashes=hashes,
                       prefix_s=prefix_s, export_s=export_s, continuation_s=continuation_s,
                       backbone_frames=counter.frames)
        save_blob(store, case, name, payload)
        return payload

    def case(self, case, store, *, measure=False, methods=METHODS):
        started = time.perf_counter()
        verify_case(case)
        if shutil.disk_usage(store.root).free < 20 * 1024**3:
            raise RuntimeError("less than 20 GiB free; stop before allocating state caches")
        frames = SharedVideoFrames(case["video_dir"], device="cuda")
        if frames.stems != case["frame_stems"]: raise ValueError("frame order changed")
        try:
            source = self.prefix(case, frames, self.small, store, "source_prefix")
            native = self.prefix(case, frames, self.base, store, "base_prefix")
            for method, payload in (("small_only", source), ("base_native", native)):
                if store.load(case, method) is None:
                    store.save(case, method, origin="executed_or_cached_prefix", scores=score_masks(case, method, payload["masks"]),
                               runtime={"scope": "prefix_reference", "prefix_s": payload["prefix_s"],
                                        "export_s": payload["export_s"], "continuation_s": payload["continuation_s"],
                                        "backbone_frames": payload["backbone_frames"]})
            for method in (GATE, *(m for m in methods if m not in ("small_only", "base_native"))):
                existing = store.load(case, method)
                if existing is not None:
                    if method == GATE and not existing.get("gate_passed"): raise RuntimeError("failed cached self-injection")
                    continue
                # Saved predictions can be rescored without rerunning GPU inference.
                predictions = load_blob(store, case, method + "_predictions")
                if predictions is None:
                    predictions = self.method(case, frames, source, native, method)
                    save_blob(store, case, method + "_predictions", predictions)
                if method == GATE:
                    if predictions["hashes"] != native["logit_hashes"]:
                        raise RuntimeError("self-injection failed: native/injected full-logit hashes differ")
                    store.save(case, method, origin="executed", gate_passed=True,
                               binary_iou=1., max_logit_error=0., runtime=predictions["runtime"])
                else:
                    store.save(case, method, origin="executed", scores=score_masks(case, method, predictions["masks"]),
                               runtime=predictions["runtime"])
            # Cost records are fresh even if accuracy was already cached.
            if measure:
                self.measure_case(case, frames, source, native, store, methods=methods)
        finally:
            frames.release()
            torch.cuda.empty_cache()
        return dict(case_id=case["case_id"], dataset=case["dataset"], length_bin=case["length_bin"],
                    seconds=time.perf_counter() - started, decode_s=frames.decode_seconds)

    def method(self, case, frames, source, native, method, *, first_only=False):
        masks, hashes = {}, {}
        export_s = native["export_s"] if method == GATE else source["export_s"] if method in self.methods else 0.
        runtime = dict(scope="test10_method_v1", model_load_s=self.model_load_s,
                       export_s=export_s, transfer_s=0., translate_s=0., inject_s=0., replay_s=0.)
        torch.cuda.reset_peak_memory_stats()
        with Counter(self.base) as counter, inference_context("bfloat16"):
            total_start = now()
            init_start = now()
            state = fresh_inference_state(self.base, frames)
            runtime["video_init_s"] = now() - init_start
            handoff_start = now()
            if method == GATE or method in self.methods:
                selected = native["state"] if method == GATE else source["state"]
                if method != GATE:
                    t = now()
                    selected = dataclasses.replace(selected, spatial_memory=selected.spatial_memory.cuda(),
                        object_pointer=selected.object_pointer.cuda(), presence_logits=selected.presence_logits.cuda())
                    runtime["transfer_s"] = now() - t
                    t = now()
                    # Match the test9 translator precision: fp32 outside predictor autocast.
                    with torch.autocast("cuda", enabled=False): selected = self.methods[method].translate(selected)
                    runtime["translate_s"] = now() - t
                t = now()
                inject_sam2_canonical_state(selected, predictor=self.base, inference_state=state)
                runtime["inject_s"] = now() - t
                runtime["handoff_bytes"] = selected.handoff_bytes()
                if counter.frames: raise RuntimeError("no-replay injection called the backbone")
            else:
                t = now()
                if method == "last_mask":
                    self.base.add_new_mask(state, frame_idx=case["switch"], obj_id=1, mask=source["last_mask"])
                    expected = [case["switch"]]
                    runtime["empty_prompt"] = not bool(source["last_mask"].any())
                else:
                    k = int(method.rsplit("_", 1)[1])
                    prompt = load_label_mask(Path(case["annotation_dir"]) / f"{case['frame_stems'][case['first']]}.png", case["object_id"])
                    self.base.add_new_mask(state, frame_idx=case["first"], obj_id=1, mask=prompt)
                    replayed = [p for p, _ in propagate(self.base, state, start=case["switch"] - k + 1, end=case["switch"])]
                    expected = [case["first"], *range(case["switch"] - k + 1, case["switch"] + 1)]
                    if replayed != expected[1:]: raise RuntimeError("wrong replay frame range")
                if counter.frames != expected: raise RuntimeError(f"unexpected replay calls {counter.frames} != {expected}")
                runtime["replay_s"] = now() - t
                runtime["handoff_bytes"] = 0  # state bytes only; RGB/prompt inputs disclosed separately
                runtime["past_rgb_frames"] = expected
            runtime["handoff_s"] = now() - handoff_start
            runtime["past_backbone_frames"] = list(counter.frames)
            t = now()
            end = case["switch"] + 1 if first_only else case["end"]
            for pos, mask in propagate(self.base, state, start=case["switch"] + 1, end=end):
                if pos == case["switch"] + 1: runtime["first_output_s"] = now() - total_start
                masks[pos] = pack((mask > 0).cpu().numpy())
                if method == GATE: hashes[pos] = logit_hash(mask)
            runtime["continuation_s"] = now() - t
            future_calls = counter.frames[len(runtime["past_backbone_frames"]):]
            if future_calls != list(range(case["switch"] + 1, end + 1)):
                raise RuntimeError(f"unexpected continuation backbone calls: {future_calls}")
            runtime["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
            runtime["switch_ready_with_export_s"] = runtime["export_s"] + runtime["video_init_s"] + runtime["handoff_s"]
        return dict(masks=masks, hashes=hashes, runtime=runtime)

    def measure_case(self, case, frames, source, native, store, *, methods=METHODS):
        """One common eager/warm-model protocol; preparation stays separately visible."""
        if store.load(case, "costs") is not None: return
        timings = {}
        # Encode timings include lazy RGB decode consistently; caches must start empty each method.
        for method in methods:
            frames.release()
            if method in ("small_only", "base_native"):
                model = self.small if method == "small_only" else self.base
                torch.cuda.reset_peak_memory_stats()
                with inference_context("bfloat16"), Counter(model) as counter:
                    start = now(); state = fresh_inference_state(model, frames)
                    prompt = load_label_mask(Path(case["annotation_dir"]) / f"{case['frame_stems'][case['first']]}.png", case["object_id"])
                    model.add_new_mask(state, frame_idx=case["first"], obj_id=1, mask=prompt)
                    for _ in propagate(model, state, start=case["first"], end=case["switch"]): pass
                    prefix_s = now() - start
                    for _ in propagate(model, state, start=case["switch"] + 1, end=case["switch"] + 1): pass
                    timings[method] = dict(scope="cold_RGB_warm_models_prefix_plus_first", prefix_s=prefix_s,
                                           first_output_s=now() - start, peak_vram_bytes=torch.cuda.max_memory_allocated(),
                                           backbone_frames=counter.frames)
                    del state
            else:
                timings[method] = self.method(case, frames, source, native, method, first_only=True)["runtime"]
                timings[method]["scope"] = "cold_RGB_warm_models_cached_source_first_output"
        store.save(case, "costs", origin="executed", timings=timings)
