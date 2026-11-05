#!/usr/bin/env python3
"""Handoff cost per object (doc 01 section 2): bytes carried, past frames reprocessed, wall time, peak GPU memory.

Time covers the handoff only: from Base+ taking over until it can process the next frame.
Frames are read into RAM first, so disk reads do not count for any method.
"""
import json
import os
import time

import numpy as np
import torch

import handoff_eval as he
from handoff_eval import (CHECKPOINTS, ROOT, CanonicalState, LazyFrames, StateSpec, fresh, inject_sam2_canonical_state,
                          load_label, precision, select_cases, video_paths)
from more_eval import MoreSources

MODELS = {'residual_mlp': 'run:spatial/mlp', 'transformer': 'run:spatial/transformer', 'mlp_context': 'run:spatial/mlp_context',
          'legacy_transformer': 'legacy', 'ridge': 'ridge', 'moment_matched': 'mm'}
FIELDS = ('spatial_memory', 'object_pointer', 'frame_indices', 'slot_order', 'is_conditioning', 'validity')
REPEAT = 3


class Preloaded:
    def __init__(self, paths, size, wanted):
        lazy = LazyFrames(paths, size)
        self.count, self.height, self.width = len(paths), lazy.height, lazy.width
        self.frames = {index: lazy[index].clone() for index in wanted}

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return self.frames[index]


def timed(function, repeat=REPEAT):
    times, peaks = [], []
    for _ in range(repeat):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        peaks.append((torch.cuda.max_memory_allocated() - before) / 2 ** 20)
    return float(np.median(times)), float(np.max(peaks))


def payload(s):
    keep = s.validity[0, 0]
    total = 16  # object id and switch frame
    for name in FIELDS:
        tensor = getattr(s, name)[0, 0]
        total += tensor[keep].numel() * tensor.element_size()
    return int(total)


def measure(row, base, sources, repeat):
    with torch.serialization.safe_globals([CanonicalState, StateSpec]):
        value = torch.load(row['path'], map_location='cpu', weights_only=True)
    s, t = value['source_canonical'], value['target_canonical']
    switch = int(s.switch_frame)
    paths, gt = video_paths(row['dataset'], row['video_id'])
    cond = (s.validity[0, 0] & s.is_conditioning[0, 0]).nonzero().flatten().tolist()
    start = int(s.frame_indices[0, 0, cond[0]])
    object_id = int(row['object_id'])
    prompt, _ = load_label(gt, paths[start], object_id)
    mask = torch.as_tensor(prompt, dtype=torch.bool)
    frames = Preloaded(paths, base.image_size, range(start, switch + 1))
    out = dict(video=row['video_id'], dataset=row['dataset'], prefix=switch - start + 1, seconds={}, peak_mb={},
               state_bytes=payload(s), prompt_rgb_bytes=os.path.getsize(paths[start]),
               prompt_mask_bytes=os.path.getsize(gt / (paths[start].stem + '.png')),
               prefix_rgb_bytes=sum(os.path.getsize(paths[f]) for f in range(start, switch + 1)))

    def translate(name):
        def run():
            with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                a, b = sources.resolve(name, s, t)
                state = s.with_continuous(spatial_memory=a[None, None].bfloat16().cpu(), object_pointer=b[None, None].float().cpu(),
                                          presence_logits=s.presence_logits.clone(), positional_information={'policy': 'regenerate_at_target'})
            with torch.inference_mode(), precision():
                inject_sam2_canonical_state(state, predictor=base, inference_state=fresh(base, frames))
        return run

    def inject_only():
        with torch.inference_mode(), precision():
            inject_sam2_canonical_state(t, predictor=base, inference_state=fresh(base, frames))

    def anchors():
        with torch.inference_mode(), precision():
            state = fresh(base, frames)
            base.add_new_mask(state, frame_idx=start, obj_id=object_id, mask=mask)
            if switch != start:
                base.add_new_mask(state, frame_idx=switch, obj_id=object_id, mask=mask)
            base.propagate_in_video_preflight(state)

    def replay(k):
        def run():
            first = start if k is None else max(start, switch - k + 1)
            with torch.inference_mode(), precision():
                state = fresh(base, frames)
                base.add_new_mask(state, frame_idx=start, obj_id=object_id, mask=mask)
                generator = base.propagate_in_video(state, start_frame_idx=first, max_frame_num_to_track=switch - first)
                for position, _, _ in generator:
                    store = state['output_dict_per_obj'][0]['non_cond_frame_outputs']
                    for old in [f for f in store if f < position - 17]:
                        del store[old]
        return run

    jobs = {f'translate:{name}': translate(source) for name, source in MODELS.items()}
    jobs['inject_only'] = inject_only
    jobs['olv'] = anchors
    for k in (4, 8, 16):
        jobs[f'replay{k}'] = replay(k)
    for name, job in jobs.items():
        out['seconds'][name], out['peak_mb'][name] = timed(job, repeat)
    out['seconds']['full_replay'], out['peak_mb']['full_replay'] = timed(replay(None), 1)
    return out


def main():
    he.AUTOCAST[0] = True
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('REVIEW_GPU_FRACTION', 0.45)))
    from sam2.build_sam import build_sam2_video_predictor
    base = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml', str(CHECKPOINTS / 'sam2.1_hiera_base_plus.pt'), device='cuda')
    sources = MoreSources(torch.device('cuda'))
    print('model MB', round(torch.cuda.memory_allocated() / 2 ** 20), flush=True)
    picks = [c for c in select_cases('val', 1, None, 7) if c['switch_frame'] < 800][:10]
    picks += select_cases('dev', 1, 20, 7, 'MOSEv2')
    measure(picks[-1], base, sources, 1)  # warm-up, not recorded
    rows = []
    for row in picks:
        rows.append(measure(row, base, sources, REPEAT))
        print(row['case_id'], rows[-1]['prefix'], round(rows[-1]['seconds']['full_replay'], 2), flush=True)
    folder = ROOT / 'runs/cost'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'cost.json').write_text(json.dumps(rows))
    names = list(rows[0]['seconds'])
    for dataset in sorted({r['dataset'] for r in rows}):
        group = [r for r in rows if r['dataset'] == dataset]
        print(f"\n{dataset}: {len(group)} cases, prefix frames median {np.median([r['prefix'] for r in group]):.0f}")
        print(f"  state bytes median {np.median([r['state_bytes'] for r in group]) / 1e6:.2f} MB; prompt jpeg {np.median([r['prompt_rgb_bytes'] for r in group]) / 1e3:.0f} KB; "
              f"prompt mask {np.median([r['prompt_mask_bytes'] for r in group]) / 1e3:.1f} KB; prefix jpegs {np.median([r['prefix_rgb_bytes'] for r in group]) / 1e6:.1f} MB")
        print('  method                       ms median   ms max   peak MB')
        for name in names:
            times = [r['seconds'][name] * 1000 for r in group]
            print(f"  {name:26s} {np.median(times):10.1f} {np.max(times):9.1f} {np.max([r['peak_mb'][name] for r in group]):9.0f}")


if __name__ == '__main__':
    main()
