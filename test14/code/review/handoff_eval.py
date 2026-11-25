#!/usr/bin/env python3
"""Memory compositions on cached LVOS/MOSE pairs, including Base+ anchor re-encoding.

Extends jf_eval: a method may also drop record groups and add Small's last visible mask,
re-encoded by Base+. Small's own run supplies that mask and the Source-only reference.

Extra method tokens (anything else is a jf_eval source spec):
  orig            Base+ conditioning record only
  lastvis         Small's last visible mask only
  olv             Base+ conditioning record + last visible mask
  olv_masks       same, built from the two masks (implementation check)
  anchor:<src>    every record from <src> + last visible mask
  anchorc:<src>   conditioning from Base+, recent from <src>, + last visible mask
  recent:<src>    recent records from <src> + last visible mask, no prompt record
"""
import argparse
import gc
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from jf_eval import LazyFrames, ROOT, Sources, assemble, fresh, parse_methods, select_cases, state_errors, video_paths
from unseen_eval import TENSORS, active_state
from vos_memory_inspector.sam2_state import canonicalize_sam2_inference_state, inject_sam2_canonical_state
from vos_memory_inspector.state_schema import CanonicalState, StateSpec
from vos_memory_inspector.vos_metrics import load_official_metrics, read_indexed_png

CHECKPOINTS = Path('/workspace/j700qtcgrg/CMMT/checkpoints')
NAN = float('nan')
# The cached pairs were built without autocast; matching it makes Small and Base+ replays bitwise exact.
AUTOCAST = [False]


def precision():
    return torch.autocast('cuda', dtype=torch.bfloat16, enabled=AUTOCAST[0])


def subset(state, keep):
    fields = {name: getattr(state, name).detach().cpu()[0, 0, keep][None, None].clone() for name in TENSORS}
    fields['slot_order'] = torch.arange(len(keep))[None, None]
    return replace(state, **fields).validate()


def load_label(gt, path, object_id):
    annotation = gt / (path.stem + '.png')
    if not annotation.is_file():
        return None, None
    label = read_indexed_png(annotation)
    return label == object_id, label == 255


def frame_scores(row, gt, path, object_id, mask, metrics):
    target, void = load_label(gt, path, object_id)
    row['area'].append(float(mask.mean()))
    if target is None:
        row['J'].append(None)
        row['F'].append(None)
        row['visible'].append(None)
    else:
        row['J'].append(float(metrics.db_eval_iou(target, mask, void)))
        row['F'].append(float(metrics.db_eval_boundary(target, mask, void)))
        row['visible'].append(bool(target.any()))


def small_track(predictor, paths, start, prompt, switch, end, object_id, gt, metrics, verify):
    """Small from the prompt to the end: Source-only scores and its last visible mask at the switch."""
    state = fresh(predictor, LazyFrames(paths, predictor.image_size))
    row = dict(J=[], F=[], visible=[], area=[])
    last, exported = (start, prompt), None
    with torch.inference_mode(), precision():
        predictor.add_new_mask(state, frame_idx=start, obj_id=1, mask=torch.as_tensor(prompt, dtype=torch.bool))
        for frame, _, logits in predictor.propagate_in_video(state, start_frame_idx=start, max_frame_num_to_track=end - start):
            mask = (logits[0, 0] > 0).cpu().numpy()
            if frame <= switch:
                memory = state['output_dict_per_obj'][0]
                entry = memory['non_cond_frame_outputs'].get(frame) or memory['cond_frame_outputs'].get(frame)
                if entry is not None and float(entry['object_score_logits'].max()) > 0 and mask.any():
                    last = (frame, mask)
                if frame == switch and verify:
                    exported = active_state(canonicalize_sam2_inference_state(state, switch_frame=frame, strict=True))
            else:
                frame_scores(row, gt, paths[frame], object_id, mask, metrics)
            store = state['output_dict_per_obj'][0]['non_cond_frame_outputs']
            for old in [f for f in store if f < frame - 17]:
                del store[old]
    return row, last, exported


def build(method, spec, sources, s, t, last, prompt_anchor):
    """(canonical or None, [(frame, mask)]) for one method."""
    token = spec[0]
    valid, cond = s.validity[0, 0], s.is_conditioning[0, 0]
    prompt_records = (valid & cond).nonzero().flatten().tolist()
    recent_records = (valid & ~cond).nonzero().flatten().tolist()
    anchor = [] if last[0] == prompt_anchor[0] else [last]
    if token == 'lastvis':
        return None, [last]
    if token == 'olv_masks':
        return None, [prompt_anchor] + anchor
    if token in ('orig', 'olv'):
        state = subset(assemble(method, ('target',) * 4, sources, s, t), prompt_records)
        return state, anchor if token == 'olv' else []
    if token.startswith('anchorc:'):
        src = token[8:]
        return assemble(method, ('target', src, 'target', src), sources, s, t), anchor
    if token.startswith('anchor:'):
        return assemble(method, (token[7:],) * 4, sources, s, t), anchor
    if token.startswith('recent:'):
        return subset(assemble(method, (token[7:],) * 4, sources, s, t), recent_records), [last]
    return assemble(method, spec, sources, s, t), []


def run_methods(predictor, paths, canonicals, switch, object_id, end, gt, metrics):
    frames = LazyFrames(paths, predictor.image_size)
    original = predictor._get_image_feature
    cache = {}
    allowed = {f for _, _, masks in canonicals for f, _ in masks}

    def feature(state, frame_idx, batch_size):
        if frame_idx <= switch and frame_idx not in allowed:
            raise RuntimeError('past RGB requested')
        if frame_idx in cache:
            state['cached_features'] = {frame_idx: cache[frame_idx]}
        result = original(state, frame_idx, batch_size)
        cache.clear()
        cache[frame_idx] = state['cached_features'][frame_idx]
        return result

    predictor._get_image_feature = feature
    output = {m: dict(J=[], F=[], visible=[], area=[]) for m, _, _ in canonicals}
    offsets, generators, states = [], {}, {}
    try:
        with torch.inference_mode(), precision():
            for method, canonical, masks in canonicals:
                state = fresh(predictor, frames)
                states[method] = state
                if canonical is not None:
                    inject_sam2_canonical_state(canonical, predictor=predictor, inference_state=state)
                for frame, mask in masks:
                    obj = state['obj_ids'][0] if state['obj_ids'] else object_id
                    predictor.add_new_mask(state, frame_idx=frame, obj_id=obj, mask=torch.as_tensor(mask, dtype=torch.bool))
                generators[method] = predictor.propagate_in_video(state, start_frame_idx=switch + 1, max_frame_num_to_track=end - switch - 1)
            for expected in range(switch + 1, end + 1):
                offsets.append(expected - switch)
                for method, generator in generators.items():
                    position, ids, logits = next(generator)
                    if position != expected or len(ids) != 1 or str(ids[0]) != str(object_id):
                        raise RuntimeError('timeline/identity mismatch')
                    store = states[method]['output_dict_per_obj'][0]['non_cond_frame_outputs']
                    for old in [f for f in store if f < position - 17]:
                        del store[old]
                    frame_scores(output[method], gt, paths[expected], object_id, (logits[0, 0] > 0).cpu().numpy(), metrics)
    finally:
        predictor._get_image_feature = original
        for generator in generators.values():
            generator.close()
    return output, offsets


def run_case(args, row, path, methods, sources, small, base, metrics, index, total):
    started = time.monotonic()
    with torch.serialization.safe_globals([CanonicalState, StateSpec]):
        value = torch.load(row['path'], map_location='cpu', weights_only=True)
    s, t = value['source_canonical'], value['target_canonical']
    switch = int(s.switch_frame)
    paths, gt = video_paths(row['dataset'], row['video_id'])
    if len(paths) != int(value['metadata']['num_frames']):
        raise ValueError('pair/video length mismatch')
    end = len(paths) - 1
    if end <= switch:
        return
    object_id = int(row['object_id'])
    cond = (s.validity[0, 0] & s.is_conditioning[0, 0]).nonzero().flatten().tolist()
    if len(cond) != 1:
        raise ValueError(f'expected one prompt record, found {len(cond)}')
    start = int(s.frame_indices[0, 0, cond[0]])
    prompt, _ = load_label(gt, paths[start], object_id)
    if prompt is None or not prompt.any():
        raise ValueError('prompt mask missing')
    if args.skip_small:
        small_row, last, exported = None, (start, prompt), None
    else:
        small_row, last, exported = small_track(small, paths, start, prompt, switch, end, object_id, gt, metrics, args.verify)
    if args.verify and exported is not None:
        same = torch.equal(exported.frame_indices, s.frame_indices) and torch.equal(exported.validity, s.validity)
        spatial = (exported.spatial_memory.float() - s.spatial_memory.float()).abs().max()
        pointer = (exported.object_pointer.float() - s.object_pointer.float()).abs().max()
        print(f'verify {row["case_id"]}: records match {same}, spatial max {float(spatial):.5f}, pointer max {float(pointer):.5f}', flush=True)
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        built = [(m, *build(m, spec, sources, s, t, last, (start, prompt))) for m, spec in methods.items()]
    errors = {m: state_errors(c, t) if c is not None and c.spatial_memory.shape == t.spatial_memory.shape and not masks else dict(spatial=NAN, pointer=NAN)
              for m, c, masks in built}
    scores, offsets = run_methods(base, paths, built, switch, object_id, end, gt, metrics)
    if small_row is not None:
        scores['small'] = small_row
        errors['small'] = dict(spatial=NAN, pointer=NAN)
        if len(small_row['J']) != len(offsets):
            raise RuntimeError('source timeline mismatch')
    path.write_text(json.dumps(dict(case=row, switch=switch, offsets=offsets, errors=errors, scores=scores, start=start,
                                    last_visible=last[0], reencoded={m: len(masks) for m, _, masks in built},
                                    seconds=time.monotonic() - started)))
    print(f"{index}/{total} {row['case_id']} frames={len(offsets)} last={last[0]}/{switch} {time.monotonic() - started:.1f}s", flush=True)
    del built, scores
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--method', action='append', required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--per-video', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--skip-small', action='store_true')
    parser.add_argument('--autocast', action='store_true')
    args = parser.parse_args()
    AUTOCAST[0] = args.autocast
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('REVIEW_GPU_FRACTION', 0.4)))
    methods = parse_methods(args.method)
    out = ROOT / 'runs/jf' / args.tag
    (out / 'cases').mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=1))
    metrics, reason = load_official_metrics()
    if metrics is None:
        raise RuntimeError(reason)
    from sam2.build_sam import build_sam2_video_predictor
    small = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml', str(CHECKPOINTS / 'sam2.1_hiera_small.pt'), device='cuda')
    base = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml', str(CHECKPOINTS / 'sam2.1_hiera_base_plus.pt'), device='cuda')
    sources = Sources(torch.device('cuda'))
    cases = select_cases(args.split, args.per_video, args.limit, args.seed, args.dataset)
    print(len(cases), 'cases', flush=True)
    failures = 0
    for index, row in enumerate(cases):
        if index % args.shards != args.shard:
            continue
        path = out / 'cases' / (hashlib.sha256(row['case_id'].encode()).hexdigest()[:24] + '.json')
        if path.exists():
            continue
        try:
            run_case(args, row, path, methods, sources, small, base, metrics, index, len(cases))
        except Exception as error:
            print(f"CASE FAILED {row['case_id']}: {type(error).__name__}: {str(error)[:300]}", flush=True)
            failures += 1
            gc.collect()
            torch.cuda.empty_cache()
    if failures:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
