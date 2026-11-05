#!/usr/bin/env python3
"""Doc baselines not measured yet, and anchors that do not trust Small's mask, on cached pairs.

Extra method tokens on top of handoff_eval:
  mm              Moment-Matched Copy (doc 01 section 3.2)
  replay:<k>      Original-Prompt(s)+Replay-k: prompt record, then Base+ tracks the last k frames before the switch
  olv_soft        prompt record + last visible mask as a recent record that expires
  pinb:<src>      conditioning from Base+, recent from <src>, + Base+'s own decode of the switch frame as a conditioning record
  pinp            prompt record + Base+'s decode of the switch frame from the prompt record alone (no translator)
  swap:<k>:<src>  full translation from <src>; once Base+ has run k frames past the switch, its own record replaces the translated conditioning record
  pin:<k>:<src>   same, but the translated conditioning record stays
  anatomy:<part>:<base>  target state plus one part of <base>'s error: near, far, low, high, shrink, rest
  band:<low>:<high>:<base>  contrast gains on <base>'s low and high spatial bands
  cal:<key>:<base>  per-channel contrast gains from runs/calib/<key>.pt

Every case also stores `probe`: Base+'s decode of the switch frame, once from the prompt record and
the translated recent records before that frame, once from the prompt record alone, and how each
agrees with Small's mask. GT scores in the probe are for diagnosis; a gate may only use the agreement.
"""
import argparse
import gc
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import handoff_eval as he
from handoff_eval import (CHECKPOINTS, NAN, ROOT, CanonicalState, LazyFrames, Sources, StateSpec, assemble, build, fresh,
                          frame_scores, inject_sam2_canonical_state, load_label, load_official_metrics, parse_methods,
                          precision, select_cases, small_track, state_errors, subset, video_paths)


class MoreSources(Sources):
    def __init__(self, device):
        super().__init__(device)
        self.moment = None
        self.prompt_mask = None
        self.gains = {}

    def anatomy(self, name, s, t):
        _, part, base = name.split(':', 2)
        a, b = self.resolve(base, s, t)
        y, q = self.resolve('target', s, t)
        error = a - y
        if part in ('near', 'far'):
            mask = F.interpolate(self.prompt_mask[None, None].float().to(self.device), size=(64, 64), mode='area')
            near = (F.max_pool2d(mask, 5, 1, 2) > 0).float()
            error = error * (near if part == 'near' else 1 - near)
        elif part in ('low', 'high'):
            low = F.interpolate(F.avg_pool2d(error, 8), size=(64, 64), mode='bilinear', align_corners=False)
            error = low if part == 'low' else error - low
        elif part in ('shrink', 'rest'):
            # Regression shrinkage: the part of the error that rescales the target around its channel mean.
            center = y - y.mean((2, 3), keepdim=True)
            shrink = (error * center).sum((2, 3), keepdim=True) / center.square().sum((2, 3), keepdim=True).clamp_min(1e-9) * center
            error = shrink if part == 'shrink' else error - shrink
        else:
            raise ValueError(name)
        return y + error, q

    def band(self, name, s, t):
        """band:<low gain>:<high gain>:<base> applies separate contrast gains to the two spatial bands."""
        _, low, high, base = name.split(':', 3)
        a, b = self.resolve(base, s, t)
        center = a.mean((2, 3), keepdim=True)
        rest = a - center
        smooth = F.interpolate(F.avg_pool2d(rest, 8), size=(64, 64), mode='bilinear', align_corners=False)
        return center + float(low) * smooth + float(high) * (rest - smooth), b

    def calibrate(self, name, s, t):
        """cal:<key>:<base> applies the per-channel gains in runs/calib/<key>.pt, by record group."""
        _, key, base = name.split(':', 2)
        a, b = self.resolve(base, s, t)
        if key not in self.gains:
            self.gains[key] = torch.load(ROOT / 'runs/calib' / f'{key}.pt', map_location=self.device, weights_only=True)
        table = self.gains[key]
        cond = s.is_conditioning[0, 0].to(self.device)
        gain = torch.where(cond[:, None], table['cond']['all'][None], table['rest']['all'][None])
        center = a.mean((2, 3), keepdim=True)
        return center + gain[:, :, None, None] * (a - center), b

    def resolve(self, name, s, t):
        if name.startswith('anatomy:'):
            return self.anatomy(name, s, t)
        if name.startswith('band:'):
            return self.band(name, s, t)
        if name.startswith('cal:'):
            return self.calibrate(name, s, t)
        if name != 'mm':
            return super().resolve(name, s, t)
        if self.moment is None:
            self.moment = torch.load(ROOT / 'runs/moment/moment.pt', map_location=self.device, weights_only=True)
        x, p = s.spatial_memory[0, 0].to(self.device).float(), s.object_pointer[0, 0].to(self.device).float()
        a, b = x.clone(), p.clone()
        cond = s.is_conditioning[0, 0].to(self.device)
        for group, mask in (('cond', cond), ('rest', ~cond)):
            if mask.any():
                m = {k: v.float()[None, :, None, None] for k, v in self.moment['spatial'][group].items()}
                a[mask] = (x[mask] - m['ms']) / m['ss'].clamp_min(1e-6) * m['st'] + m['mt']
                q = {k: v.float()[None] for k, v in self.moment['pointer'][group].items()}
                b[mask] = (p[mask] - q['ms']) / q['ss'].clamp_min(1e-6) * q['st'] + q['mt']
        return a, b


def iou(a, b):
    union = (a | b).sum()
    return 1.0 if union == 0 else float((a & b).sum() / union)


def decode_switch(predictor, frames, canonical, switch):
    """Base+'s mask on the switch frame, from records strictly before it."""
    state = fresh(predictor, frames)
    with torch.inference_mode(), precision():
        inject_sam2_canonical_state(canonical, predictor=predictor, inference_state=state)
        generator = predictor.propagate_in_video(state, start_frame_idx=switch, max_frame_num_to_track=0)
        position, _, logits = next(generator)
        generator.close()
    if position != switch:
        raise RuntimeError('probe timeline mismatch')
    return (logits[0, 0] > 0).cpu().numpy()


def probe(predictor, paths, sources, s, t, switch, last, gt, object_id, metrics, source):
    frames = LazyFrames(paths, predictor.image_size)
    valid = s.validity[0, 0]
    prompt_records = (valid & s.is_conditioning[0, 0]).nonzero().flatten().tolist()
    earlier = (valid & (s.frame_indices[0, 0] != switch)).nonzero().flatten().tolist()
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        full = assemble('probe', ('target', source, 'target', source), sources, s, t)
        translated, prompt_only = subset(full, earlier), subset(full, prompt_records)
    decoded = decode_switch(predictor, frames, translated, switch)
    alone = decode_switch(predictor, frames, prompt_only, switch)
    small_mask = last[1] if last[0] == switch else np.zeros_like(decoded)
    target, void = load_label(gt, paths[switch], object_id)
    row = dict(iou=iou(decoded, small_mask), iou_prompt=iou(alone, small_mask), area_base=float(decoded.mean()),
               area_prompt=float(alone.mean()), area_small=float(small_mask.mean()), last_gap=switch - last[0])
    if target is not None and target.any():
        row.update(j_base=float(metrics.db_eval_iou(target, decoded, void)), j_prompt=float(metrics.db_eval_iou(target, alone, void)),
                   j_small=float(metrics.db_eval_iou(target, small_mask, void)))
    return row, decoded, alone


def plan(method, spec, sources, s, t, last, prompt_anchor, switch, decoded, alone):
    row = dict(method=method, canonical=None, masks=[], replay_from=None, soft=None, refresh=None)
    token = spec[0]
    if token.startswith('replay:'):
        row.update(masks=[prompt_anchor], replay_from=max(prompt_anchor[0], switch - int(token[7:]) + 1))
    elif token == 'olv_soft':
        canonical, masks = build(method, ('olv',) * 4, sources, s, t, last, prompt_anchor)
        row.update(canonical=canonical, masks=masks, soft=masks[0][0] if masks else None)
    elif token.startswith(('swap:', 'pin:')):
        mode, k, source = token.split(':', 2)
        row.update(canonical=assemble(method, (source,) * 4, sources, s, t), refresh=dict(mode=mode, k=int(k), done=False, frame=None))
    elif token == 'pinp':
        canonical, _ = build(method, ('orig',) * 4, sources, s, t, last, prompt_anchor)
        row.update(canonical=canonical, masks=[(switch, alone)] if alone.any() and switch != prompt_anchor[0] else [])
    elif token.startswith('pinb:'):
        source = token[5:]
        row.update(canonical=assemble(method, ('target', source, 'target', source), sources, s, t),
                   masks=[(switch, decoded)] if decoded.any() else [])
    else:
        canonical, masks = build(method, spec, sources, s, t, last, prompt_anchor)
        row.update(canonical=canonical, masks=masks)
    return row


def refresh(state, position, switch, mask, rule):
    """Base+'s own record of a frame it just processed becomes a conditioning record. No RGB beyond that frame."""
    if position < switch + rule['k'] or not mask.any():
        return
    store = state['output_dict_per_obj'][0]
    record = store['non_cond_frame_outputs'].get(position)
    if record is None or float(record['object_score_logits'].max()) <= 0:
        return
    if rule['mode'] == 'swap':
        for frame in [f for f in store['cond_frame_outputs'] if f <= switch]:
            del store['cond_frame_outputs'][frame]
    store['cond_frame_outputs'][position] = store['non_cond_frame_outputs'].pop(position)
    rule['done'], rule['frame'] = True, position


def run_all(predictor, paths, plans, switch, object_id, end, gt, metrics):
    frames = LazyFrames(paths, predictor.image_size)
    original = predictor._get_image_feature
    cache = {}
    allowed = set()
    for row in plans:
        allowed |= {f for f, _ in row['masks']}
        if row['replay_from'] is not None:
            allowed |= set(range(row['replay_from'], switch + 1))

    def feature(state, frame_idx, batch_size):
        if frame_idx <= switch and frame_idx not in allowed:
            raise RuntimeError('past RGB requested')
        if frame_idx in cache:
            state['cached_features'] = {frame_idx: cache[frame_idx]}
        result = original(state, frame_idx, batch_size)
        cache.clear()
        cache[frame_idx] = state['cached_features'][frame_idx]
        return result

    def prune(state, position):
        store = state['output_dict_per_obj'][0]['non_cond_frame_outputs']
        for old in [f for f in store if f < position - 17]:
            del store[old]

    predictor._get_image_feature = feature
    output = {row['method']: dict(J=[], F=[], visible=[], area=[]) for row in plans}
    offsets, generators, states = [], {}, {}
    try:
        with torch.inference_mode(), precision():
            for row in plans:
                method = row['method']
                state = fresh(predictor, frames)
                states[method] = state
                if row['canonical'] is not None:
                    inject_sam2_canonical_state(row['canonical'], predictor=predictor, inference_state=state)
                for frame, mask in row['masks']:
                    obj = state['obj_ids'][0] if state['obj_ids'] else object_id
                    predictor.add_new_mask(state, frame_idx=frame, obj_id=obj, mask=torch.as_tensor(mask, dtype=torch.bool))
                if row['soft'] is not None:
                    # A recent record is read by frame index, so an old mask is filed at the switch frame.
                    predictor.propagate_in_video_preflight(state)
                    store = state['output_dict_per_obj'][0]
                    store['non_cond_frame_outputs'][switch] = store['cond_frame_outputs'].pop(row['soft'])
                first = switch + 1 if row['replay_from'] is None else row['replay_from']
                generator = predictor.propagate_in_video(state, start_frame_idx=first, max_frame_num_to_track=end - first)
                for expected in range(first, switch + 1):
                    position, _, _ = next(generator)
                    if position != expected:
                        raise RuntimeError('replay timeline mismatch')
                    prune(state, position)
                generators[method] = generator
            rules = {row['method']: row['refresh'] for row in plans}
            for expected in range(switch + 1, end + 1):
                offsets.append(expected - switch)
                for method, generator in generators.items():
                    position, ids, logits = next(generator)
                    if position != expected or len(ids) != 1 or str(ids[0]) != str(object_id):
                        raise RuntimeError('timeline/identity mismatch')
                    prune(states[method], position)
                    mask = (logits[0, 0] > 0).cpu().numpy()
                    frame_scores(output[method], gt, paths[expected], object_id, mask, metrics)
                    if rules[method] is not None and not rules[method]['done']:
                        refresh(states[method], position, switch, mask, rules[method])
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
    sources.prompt_mask = torch.as_tensor(prompt)
    last, gate, decoded, alone = (start, prompt), None, None, None
    if not args.skip_small:
        _, last, _ = small_track(small, paths, start, prompt, switch, switch, object_id, gt, metrics, False)
        gate, decoded, alone = probe(base, paths, sources, s, t, switch, last, gt, object_id, metrics, args.probe_source)
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        plans = [plan(m, spec, sources, s, t, last, (start, prompt), switch, decoded, alone) for m, spec in methods.items()]
    errors = {p['method']: state_errors(p['canonical'], t)
              if p['canonical'] is not None and p['canonical'].spatial_memory.shape == t.spatial_memory.shape and not p['masks']
              else dict(spatial=NAN, pointer=NAN) for p in plans}
    scores, offsets = run_all(base, paths, plans, switch, object_id, end, gt, metrics)
    path.write_text(json.dumps(dict(case=row, switch=switch, offsets=offsets, errors=errors, scores=scores, start=start, last_visible=last[0],
                                    probe=gate, reencoded={p['method']: len(p['masks']) + (0 if p['replay_from'] is None else switch - p['replay_from'] + 1)
                                                           for p in plans},
                                    refreshed={p['method']: p['refresh']['frame'] - switch if p['refresh'] and p['refresh']['frame'] else None for p in plans},
                                    seconds=time.monotonic() - started)))
    print(f"{index}/{total} {row['case_id']} frames={len(offsets)} last={last[0]}/{switch} {time.monotonic() - started:.1f}s", flush=True)
    del plans, scores
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
    parser.add_argument('--autocast', action='store_true')
    parser.add_argument('--probe-source', default='ridge')
    parser.add_argument('--skip-small', action='store_true')
    parser.add_argument('--only')
    args = parser.parse_args()
    he.AUTOCAST[0] = args.autocast
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
    sources = MoreSources(torch.device('cuda'))
    cases = select_cases(args.split, args.per_video, args.limit, args.seed, args.dataset)
    print(len(cases), 'cases', flush=True)
    failures = 0
    for index, row in enumerate(cases):
        if index % args.shards != args.shard or (args.only and row['video_id'] not in args.only.split(',')):
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
