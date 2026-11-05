#!/usr/bin/env python3
"""Live handoff on datasets never used for training: Small rollout, translate, Base+ continuation.

Same case rules as the earlier benchmark: object start = first visible GT frame, end = last
video frame, switch = start + fraction*(end-start), scored on GT-visible frames after the switch.
Reference runs are Small alone and Base+ from the object's first frame (Full Replay).
"""
import argparse
import gc
import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from jf_eval import ROOT, Sources, assemble, fresh, parse_methods
from vos_memory_inspector.sam2_state import canonicalize_sam2_inference_state, inject_sam2_canonical_state
from vos_memory_inspector.vos_metrics import load_official_metrics

DATA = Path('/workspace/j700qtcgrg/CMMT/data')
CHECKPOINTS = Path('/workspace/j700qtcgrg/CMMT/checkpoints')
TENSORS = ('spatial_memory', 'object_pointer', 'presence_logits', 'frame_indices', 'slot_order', 'is_conditioning', 'validity')


def natural(path):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', path.stem)]


def read_label(path):
    with Image.open(path) as image:
        if image.mode in ('P', 'L', 'I', 'I;16'):
            return np.array(image)
        if image.mode == '1':
            return np.array(image).astype(np.uint8)
        rgb = np.array(image.convert('RGB')).astype(np.int64)
        return rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]


class Video:
    def __init__(self, dataset, name):
        if dataset == 'm3vos':
            base = DATA / 'M3VOS-manual/data'
            self.ignore = 255
        elif dataset == 'pumavos':
            base = DATA / 'PUMaVOS/extracted/PUBLIC_PUMaVOS'
            self.ignore = None
        else:
            raise ValueError(dataset)
        self.dataset, self.name = dataset, name
        self.frames = sorted([p for p in (base / 'JPEGImages' / name).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')], key=natural)
        index = {p.stem: i for i, p in enumerate(self.frames)}
        self.masks = {index[p.stem]: p for p in (base / 'Annotations' / name).iterdir() if p.suffix == '.png' and p.stem in index}
        with Image.open(self.masks[min(self.masks)]) as image:
            self.height, self.width = image.height, image.width

    def labels(self, frame):
        path = self.masks.get(frame)
        if path is None:
            return None
        label = read_label(path)
        void = label == self.ignore if self.ignore is not None else np.zeros(label.shape, dtype=bool)
        return label, void

    def objects(self):
        first = {}
        for frame in sorted(self.masks):
            label, void = self.labels(frame)
            for value in np.unique(label[~void]):
                if value != 0:
                    first.setdefault(int(value), frame)
        end = len(self.frames) - 1
        return [(obj, start, end) for obj, start in sorted(first.items()) if end - start >= 8]


class Frames:
    MEAN = torch.tensor([.485, .456, .406])[:, None, None]
    STD = torch.tensor([.229, .224, .225])[:, None, None]

    def __init__(self, video, size):
        self.paths, self.size = video.frames, size
        self.height, self.width = video.height, video.width
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if index not in self.cache:
            with Image.open(self.paths[index]) as image:
                data = image.convert('RGB').resize((self.size, self.size))
                x = torch.from_numpy(np.asarray(data, dtype=np.float32).copy() / 255).permute(2, 0, 1)
            self.cache[index] = (x - self.MEAN) / self.STD
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        return self.cache[index]


def active_state(state, num_maskmem=7, max_obj_ptrs=16):
    """All conditioning + latest non-conditioning records; same rule as the pair caches."""
    limit = max(num_maskmem - 1, max_obj_ptrs - 1)
    valid = state.validity[0, 0].nonzero().flatten().tolist()
    valid.sort(key=lambda i: int(state.frame_indices[0, 0, i]))
    cond = [i for i in valid if bool(state.is_conditioning[0, 0, i])]
    recent = [i for i in valid if not bool(state.is_conditioning[0, 0, i])]
    selected = set(cond + recent[-limit:])
    keep = [i for i in valid if i in selected]
    fields = {}
    for name in TENSORS:
        old = getattr(state, name).detach().cpu()
        fields[name] = old[0, 0, keep][None, None].clone()
    fields['slot_order'] = torch.arange(len(keep))[None, None]
    metadata = {k: v for k, v in state.metadata.items() if k != 'preserved_pred_masks'}
    return replace(state, **fields, metadata=metadata).validate()


def score(video, frame, mask, obj, metrics):
    item = video.labels(frame)
    if item is None:
        return None
    label, void = item
    target = (label == obj) & ~void
    return (float(metrics.db_eval_iou(target, mask, void)), float(metrics.db_eval_boundary(target, mask, void)), bool(target.any()))


def rollout(predictor, video, obj, start, end, switches, metrics):
    """Track from the prompt to the end; export the active state at each switch."""
    frames = Frames(video, predictor.image_size)
    state = fresh(predictor, frames)
    prompt = video.labels(start)[0] == obj
    scores, exported, anchors = {}, {}, {}
    last = (start, prompt)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        predictor.add_new_mask(state, frame_idx=start, obj_id=1, mask=torch.as_tensor(prompt, dtype=torch.bool))
        for frame, ids, logits in predictor.propagate_in_video(state, start_frame_idx=start, max_frame_num_to_track=end - start):
            mask = (logits[0, 0] > 0).cpu().numpy()
            value = score(video, frame, mask, obj, metrics)
            if value is not None:
                scores[frame] = value
            memory = state['output_dict_per_obj'][0]
            entry = memory['non_cond_frame_outputs'].get(frame) or memory['cond_frame_outputs'].get(frame)
            if entry is not None and float(entry['object_score_logits'].max()) > 0 and mask.any():
                last = (frame, mask)
            if frame in switches:
                exported[frame] = active_state(canonicalize_sam2_inference_state(state, switch_frame=frame, strict=True))
                anchors[frame] = last
            # Older records are never read again; dropping them keeps long videos small.
            store = state['output_dict_per_obj'][0]['non_cond_frame_outputs']
            for old in [f for f in store if f < frame - 17]:
                del store[old]
    return scores, exported, anchors


def continuations(predictor, video, canonicals, switch, obj, end, metrics, after=None):
    frames = Frames(video, predictor.image_size)
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
    output = {m: {} for m, _, _ in canonicals}
    generators = {}
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for method, canonical, masks in canonicals:
                state = fresh(predictor, frames)
                if canonical is not None:
                    inject_sam2_canonical_state(canonical, predictor=predictor, inference_state=state)
                for frame, mask in masks:
                    predictor.add_new_mask(state, frame_idx=frame, obj_id=1, mask=torch.as_tensor(mask, dtype=torch.bool))
                generators[method] = (state, predictor.propagate_in_video(state, start_frame_idx=switch + 1, max_frame_num_to_track=end - switch - 1))
            for expected in range(switch + 1, end + 1):
                for method, (state, generator) in generators.items():
                    position, ids, logits = next(generator)
                    if position != expected:
                        raise RuntimeError('timeline mismatch')
                    mask = (logits[0, 0] > 0).cpu().numpy()
                    value = score(video, position, mask, obj, metrics)
                    if value is not None:
                        output[method][position] = value
                    if after is not None:
                        after(method, state, position, mask)
                    store = state['output_dict_per_obj'][0]['non_cond_frame_outputs']
                    for old in [f for f in store if f < position - 17]:
                        del store[old]
    finally:
        predictor._get_image_feature = original
        for _, generator in generators.values():
            generator.close()
    return output


def summarize(scores, frames):
    rows = [scores[f] for f in frames if f in scores and scores[f][2]]
    if not rows:
        return dict(J=None, F=None, n=0)
    j = sum(r[0] for r in rows) / len(rows)
    f = sum(r[1] for r in rows) / len(rows)
    return dict(J=j, F=f, JF=(j + f) / 2, n=len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--method', action='append', required=True)
    parser.add_argument('--fractions', default='0.5')
    parser.add_argument('--max-videos', type=int)
    parser.add_argument('--names')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('REVIEW_GPU_FRACTION', 0.4)))
    methods = parse_methods(args.method)
    fractions = [float(x) for x in args.fractions.split(',')]
    out = ROOT / 'runs/unseen' / args.tag
    (out / 'cases').mkdir(parents=True, exist_ok=True)
    metrics, reason = load_official_metrics()
    if metrics is None:
        raise RuntimeError(reason)
    from sam2.build_sam import build_sam2_video_predictor
    small = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml', str(CHECKPOINTS / 'sam2.1_hiera_small.pt'), device='cuda')
    base = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml', str(CHECKPOINTS / 'sam2.1_hiera_base_plus.pt'), device='cuda')
    device = torch.device('cuda')
    sources = Sources(device)
    if args.dataset == 'm3vos':
        names = [x.strip() for x in (DATA / 'M3VOS-manual/data/ImageSets/val.txt').read_text().splitlines() if x.strip()]
    else:
        names = sorted(p.name for p in (DATA / 'PUMaVOS/extracted/PUBLIC_PUMaVOS/JPEGImages').iterdir() if p.is_dir())
    if args.max_videos:
        names = names[:args.max_videos]
    if args.names:
        names = args.names.split(',')
    failures = 0
    for position, name in enumerate(names):
        if position % args.shards != args.shard:
            continue
        video = Video(args.dataset, name)
        for obj, start, end in video.objects():
            path = out / 'cases' / f'{name}_{obj}.json'
            if path.exists():
                continue
            try:
                run_case(args, video, name, obj, start, end, path, fractions, methods, sources, small, base, metrics)
            except Exception as error:
                print(f'CASE FAILED {name} {obj}: {type(error).__name__}: {str(error)[:300]}', flush=True)
                failures += 1
                gc.collect()
                torch.cuda.empty_cache()
    if failures:
        raise SystemExit(3)


def run_case(args, video, name, obj, start, end, path, fractions, methods, sources, small, base, metrics):
    if True:
        if True:
            started = time.monotonic()
            points = {}
            for fraction in fractions:
                frame = min(max(start + round(fraction * (end - start)), start + 1), end - 1)
                points[str(round(fraction * 100))] = frame
            small_scores, source_states, anchors = rollout(small, video, obj, start, end, set(points.values()), metrics)
            base_scores, target_states, _ = rollout(base, video, obj, start, end, set(points.values()), metrics)
            result = dict(dataset=args.dataset, video=name, object=obj, start=start, end=end, switches={})
            for label, switch in points.items():
                s, t = source_states[switch], target_states[switch]
                if not torch.equal(s.validity, t.validity) or not torch.equal(s.frame_indices, t.frame_indices):
                    # Records differ only if a prompt/timeline bug exists; refuse silently mixing them.
                    raise RuntimeError('source/target record mismatch')
                with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                    canonicals = []
                    last_frame, last_mask = anchors[switch]
                    for m, spec in methods.items():
                        if spec[0] == 'orig_lastvis':
                            prompt = video.labels(start)[0] == obj
                            masks = [(start, prompt)] + ([(last_frame, last_mask)] if last_frame != start else [])
                            canonicals.append((m, None, masks))
                        elif spec[0].startswith('anchorc:'):
                            src = spec[0][8:]
                            canonicals.append((m, assemble(m, ('target', src, 'target', src), sources, s, t), [(last_frame, last_mask)]))
                        elif spec[0].startswith('anchor:'):
                            canonicals.append((m, assemble(m, (spec[0][7:],) * 4, sources, s, t), [(last_frame, last_mask)]))
                        else:
                            canonicals.append((m, assemble(m, spec, sources, s, t), []))
                scored = continuations(base, video, canonicals, switch, obj, end, metrics)
                frames = list(range(switch + 1, end + 1))
                row = dict(switch=switch, last_visible=last_frame, small=summarize(small_scores, frames), base=summarize(base_scores, frames))
                row['trace'] = {m: [v.get(f, (None, None, None))[0] for f in frames] for m, v in dict(scored, small=small_scores, base=base_scores).items()}
                first30 = frames[:30]
                row['small_first30'] = summarize(small_scores, first30)
                row['base_first30'] = summarize(base_scores, first30)
                for method, values in scored.items():
                    row[method] = summarize(values, frames)
                    row[method + '_first30'] = summarize(values, first30)
                result['switches'][label] = row
            result['seconds'] = time.monotonic() - started
            path.write_text(json.dumps(result))
            line = result['switches'][next(iter(points))]
            gc.collect()
            torch.cuda.empty_cache()
            print(name, obj, {k: round(v['JF'], 3) for k, v in line.items() if isinstance(v, dict) and v.get('JF') is not None and 'first30' not in k and k != 'trace'},
                  f"{result['seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
