#!/usr/bin/env python3
"""Interleaved Base+ continuations for several state variants of one case.

Each method assembles (conditioning spatial, recent spatial, conditioning pointer,
recent pointer) from named sources, so the J&F cost of each error can be isolated.
The Base+ image encoder runs once per frame and is shared; temporal states stay
independent, as in the earlier affine evaluation.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vos_memory_inspector.sam2_state import inject_sam2_canonical_state
from vos_memory_inspector.state_schema import CanonicalState, StateSpec
from vos_memory_inspector.state_training.trainer import restore_model
from vos_memory_inspector.vos_metrics import load_official_metrics, read_indexed_png

ROOT = Path(os.environ['REVIEW_ROOT'])
DATA = Path('/workspace/j700qtcgrg/CMMT/data')
LEGACY = Path('/workspace/j700qtcgrg/CMMT-official-isolated/official-20261005T012256KST/'
              'final-state-loss-20261005T1627KST/delivery/selected_state_loss_best/translator_weights.pth')
SCALES = json.loads((ROOT / 'data/normalization.json').read_text())['scales']


class LazyFrames:
    """Same pinned SAM2 JPEG resize/normalization, without whole-video RAM use."""

    def __init__(self, paths, size):
        self.paths, self.size = paths, size
        with Image.open(paths[0]) as image:
            self.width, self.height = image.size
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from sam2.utils.misc import _load_img_as_tensor
        if index not in self.cache:
            image, h, w = _load_img_as_tensor(str(self.paths[index]), self.size)
            image = image.float()
            image.sub_(torch.tensor([.485, .456, .406])[:, None, None])
            image.div_(torch.tensor([.229, .224, .225])[:, None, None])
            self.cache[index] = image
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        return self.cache[index]


def fresh(predictor, frames):
    return dict(images=frames, num_frames=len(frames), offload_video_to_cpu=True,
                offload_state_to_cpu=True, video_height=frames.height, video_width=frames.width,
                device=predictor.device, storage_device=torch.device('cpu'),
                point_inputs_per_obj={}, mask_inputs_per_obj={}, cached_features={}, constants={},
                obj_id_to_idx=OrderedDict(), obj_idx_to_id=OrderedDict(), obj_ids=[],
                output_dict_per_obj={}, temp_output_dict_per_obj={}, frames_tracked_per_obj={})


def video_paths(dataset, video_id):
    name = 'MOSEv2' if dataset == 'MOSEv2' else 'LVOSv2'
    matches = [root for root in (DATA / name / 'extracted').iterdir()
               if (root / 'JPEGImages' / video_id).is_dir() and (root / 'Annotations' / video_id).is_dir()]
    if len(matches) != 1:
        raise ValueError(f'ambiguous/missing video {video_id}')
    rgb, gt = matches[0] / 'JPEGImages' / video_id, matches[0] / 'Annotations' / video_id
    paths = sorted([p for p in rgb.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg')], key=lambda p: int(p.stem))
    return paths, gt


def shared_continuations(predictor, frames, canonicals, switch, object_id, end, gt, paths, metrics):
    original_feature, original_forward = predictor._get_image_feature, predictor.forward_image
    cache, calls, current = {}, [], [None]

    def feature(state, frame_idx, batch_size):
        if frame_idx <= switch:
            raise RuntimeError('past RGB requested')
        current[0] = int(frame_idx)
        if frame_idx in cache:
            state['cached_features'] = {frame_idx: cache[frame_idx]}
        result = original_feature(state, frame_idx, batch_size)
        cache.clear()
        cache[frame_idx] = state['cached_features'][frame_idx]
        return result

    def forward(image):
        calls.append(current[0])
        return original_forward(image)
    predictor._get_image_feature, predictor.forward_image = feature, forward
    output = {m: dict(J=[], F=[], visible=[], area=[]) for m, _ in canonicals}
    offsets = []
    generators, states = {}, {}
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for method, canonical in canonicals:
                state = fresh(predictor, frames)
                states[method] = state
                inject_sam2_canonical_state(canonical, predictor=predictor, inference_state=state)
                generators[method] = predictor.propagate_in_video(
                    state, start_frame_idx=switch + 1, max_frame_num_to_track=end - switch - 1)
            if calls:
                raise RuntimeError('injection ran the backbone')
            for expected in range(switch + 1, end + 1):
                annotation = gt / (paths[expected].stem + '.png')
                label = read_indexed_png(annotation) if annotation.is_file() else None
                target = label == object_id if label is not None else None
                void = label == 255 if label is not None else None
                offsets.append(expected - switch)
                for method, generator in generators.items():
                    position, ids, logits = next(generator)
                    if position != expected or len(ids) != 1 or str(ids[0]) != str(object_id):
                        raise RuntimeError('timeline/identity mismatch')
                    # Records older than the read window are never used again; keep long videos small.
                    store = states[method]['output_dict_per_obj'][0]['non_cond_frame_outputs']
                    for old in [f for f in store if f < position - 17]:
                        del store[old]
                    mask = (logits[0, 0] > 0).cpu().numpy()
                    item = output[method]
                    item['area'].append(float(mask.mean()))
                    if label is None:
                        item['J'].append(None)
                        item['F'].append(None)
                        item['visible'].append(None)
                    else:
                        item['J'].append(float(metrics.db_eval_iou(target, mask, void)))
                        item['F'].append(float(metrics.db_eval_boundary(target, mask, void)))
                        item['visible'].append(bool(target.any()))
    finally:
        predictor._get_image_feature, predictor.forward_image = original_feature, original_forward
        for generator in generators.values():
            generator.close()
    return output, offsets, len(calls)


class Sources:
    """Per-case record tensors [K,...] on the device, computed once per named source."""

    def __init__(self, device):
        self.device = device
        self.legacy = None
        self.ridge = None
        self.runs = {}
        self.mean_pointer = None

    def resolve(self, name, s, t):
        x, p = s.spatial_memory[0, 0].to(self.device).float(), s.object_pointer[0, 0].to(self.device).float()
        if name == 'target':
            return t.spatial_memory[0, 0].to(self.device).float(), t.object_pointer[0, 0].to(self.device).float()
        if name == 'direct':
            return x, p
        if name == 'legacy':
            if self.legacy is None:
                raw = torch.load(LEGACY, map_location='cpu', weights_only=True)
                self.legacy = restore_model('transformer', raw.get('transformer', raw)).to(self.device).float().eval()
            a, b = self.legacy.translate_tensors(x[None, None], p[None, None])
            return a[0, 0], b[0, 0]
        if name == 'ridge':
            if self.ridge is None:
                self.ridge = torch.load(ROOT / 'runs/ridge/ridge.pt', map_location=self.device, weights_only=True)
            r = self.ridge
            return x.movedim(1, -1).double().matmul(r['spatial_w']).add(r['spatial_b']).movedim(-1, 1).float(), \
                p.double().matmul(r['pointer_w']).add(r['pointer_b']).float()
        if name == 'meanptr':
            if self.mean_pointer is None:
                self.ridge = torch.load(ROOT / 'runs/ridge/ridge.pt', map_location=self.device, weights_only=True)
                self.mean_pointer = self.ridge['target_pointer_mean'].float()
            return x, self.mean_pointer.expand_as(p)
        if name.startswith('blend:'):
            # blend:<factor>:<base> scales the base method's error around the target state.
            _, factor, base = name.split(':', 2)
            a, b = self.resolve(base, s, t)
            y, q = self.resolve('target', s, t)
            return y + float(factor) * (a - y), q + float(factor) * (b - q)
        if name.startswith('noise:'):
            # noise:<factor>:<base> keeps each record's error energy but removes its structure.
            _, factor, base = name.split(':', 2)
            a, b = self.resolve(base, s, t)
            y, q = self.resolve('target', s, t)
            generator = torch.Generator(device=self.device).manual_seed(int(s.switch_frame) + 7)
            sigma_s = (a - y).square().mean((1, 2, 3), keepdim=True).sqrt()
            sigma_p = (b - q).square().mean(1, keepdim=True).sqrt()
            return y + float(factor) * sigma_s * torch.randn(y.shape, device=self.device, generator=generator), \
                q + float(factor) * sigma_p * torch.randn(q.shape, device=self.device, generator=generator)
        if name.startswith('rescale:'):
            # rescale:<gain>:<base> undoes regression shrinkage around the per-record channel mean.
            _, gain, base = name.split(':', 2)
            a, b = self.resolve(base, s, t)
            center = a.mean((2, 3), keepdim=True)
            return center + float(gain) * (a - center), b
        if name.startswith('run:'):
            key = name[4:]
            if key not in self.runs:
                from review_models import load_run
                self.runs[key] = load_run(ROOT / 'runs' / key, self.device)
            return self.runs[key](x, p, s.is_conditioning[0, 0].to(self.device))
        raise ValueError(name)


def assemble(method, spec, sources, s, t):
    sp_cond, sp_rest, ptr_cond, ptr_rest = spec
    cond = s.is_conditioning[0, 0].to(sources.device)
    a, _ = sources.resolve(sp_rest, s, t)
    a = a.clone()
    if cond.any():
        a[cond] = sources.resolve(sp_cond, s, t)[0][cond]
    _, b = sources.resolve(ptr_rest, s, t)
    b = b.clone()
    if cond.any():
        b[cond] = sources.resolve(ptr_cond, s, t)[1][cond]
    return s.with_continuous(spatial_memory=a[None, None].bfloat16().cpu(), object_pointer=b[None, None].float().cpu(),
                             presence_logits=s.presence_logits.clone(),
                             positional_information={'policy': 'regenerate_at_target'})


def state_errors(state, t):
    cond = t.is_conditioning[0, 0]
    x, y = state.spatial_memory[0, 0].float(), t.spatial_memory[0, 0].float()
    p, q = state.object_pointer[0, 0], t.object_pointer[0, 0]
    row = dict(spatial=float((x - y).square().mean() / SCALES['spatial'] ** 2),
               pointer=float((p - q).square().mean() / SCALES['pointer'] ** 2))
    if cond.any():
        row['spatial_cond'] = float((x[cond] - y[cond]).square().mean() / SCALES['spatial'] ** 2)
        row['pointer_cond'] = float((p[cond] - q[cond]).square().mean() / SCALES['pointer'] ** 2)
    return row


def select_cases(split, per_video, limit, seed, dataset=None):
    meta = torch.load(ROOT / 'data' / split / 'meta.pt', map_location='cpu', weights_only=False)
    pool = [c for c in meta['cases'] if dataset is None or c['dataset'] == dataset]
    rows = sorted(pool, key=lambda c: hashlib.sha256(f"{seed}|{c['video_id']}|{c['case_id']}".encode()).hexdigest())
    chosen, count = [], {}
    for row in rows:
        key = row['video_id']
        if count.get(key, 0) >= per_video:
            continue
        count[key] = count.get(key, 0) + 1
        chosen.append(row)
        if limit and len(chosen) >= limit:
            break
    return chosen


def parse_methods(values):
    methods = {}
    for value in values:
        name, spec = value.split('=')
        parts = spec.split(',')
        methods[name] = (parts * 4)[:4] if len(parts) == 1 else tuple(parts)
    return methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--method', action='append', required=True, help='name=source or name=sp_cond,sp_rest,ptr_cond,ptr_rest')
    parser.add_argument('--dataset')
    parser.add_argument('--per-video', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--max-future', type=int)
    args = parser.parse_args()
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('REVIEW_GPU_FRACTION', 0.4)))
    methods = parse_methods(args.method)
    out = ROOT / 'runs/jf' / args.tag
    (out / 'cases').mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=1))
    metrics, reason = load_official_metrics()
    if metrics is None:
        raise RuntimeError(reason)
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml',
                                           '/workspace/j700qtcgrg/CMMT/checkpoints/sam2.1_hiera_base_plus.pt', device='cuda')
    sources = Sources(device)
    cases = select_cases(args.split, args.per_video, args.limit, args.seed, args.dataset)
    print(len(cases), 'cases', flush=True)
    failures = 0
    for index, row in enumerate(cases):
        if index % args.shards != args.shard:
            continue
        name = hashlib.sha256(row['case_id'].encode()).hexdigest()[:24]
        path = out / 'cases' / f'{name}.json'
        if path.exists():
            continue
        try:
            run_case(args, row, path, methods, sources, predictor, metrics, index, len(cases))
        except Exception as error:
            print(f"CASE FAILED {row['case_id']}: {type(error).__name__}: {str(error)[:300]}", flush=True)
            failures += 1
            gc.collect()
            torch.cuda.empty_cache()
    if failures:
        raise SystemExit(3)


def run_case(args, row, path, methods, sources, predictor, metrics, index, total):
    if True:
        started = time.monotonic()
        with torch.serialization.safe_globals([CanonicalState, StateSpec]):
            value = torch.load(row['path'], map_location='cpu', weights_only=True)
        s, t = value['source_canonical'], value['target_canonical']
        switch = int(s.switch_frame)
        paths, gt = video_paths(row['dataset'], row['video_id'])
        if len(paths) != int(value['metadata']['num_frames']):
            raise ValueError('pair/video length mismatch')
        end = len(paths) - 1 if args.max_future is None else min(len(paths) - 1, switch + args.max_future)
        if end <= switch:
            return
        frames = LazyFrames(paths, predictor.image_size)
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            canonicals = [(m, assemble(m, spec, sources, s, t)) for m, spec in methods.items()]
        errors = {m: state_errors(c, t) for m, c in canonicals}
        scores, offsets, calls = shared_continuations(predictor, frames, canonicals, switch, int(row['object_id']), end, gt, paths, metrics)
        path.write_text(json.dumps(dict(case=row, switch=switch, offsets=offsets, errors=errors, scores=scores,
                                        records=int(s.validity.sum()), seconds=time.monotonic() - started, backbone_calls=calls)))
        print(f"{index}/{total} {row['case_id']} frames={len(offsets)} {time.monotonic() - started:.1f}s "
              f"alloc={torch.cuda.memory_allocated() / 2 ** 30:.2f}G reserved={torch.cuda.memory_reserved() / 2 ** 30:.2f}G", flush=True)
        del canonicals, scores
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
