#!/usr/bin/env python3
"""One method per process, alone on one GPU: switch cost with the same start and end point for every method.

Start: the tracker that ran up to s still holds its raw SAM2 inference state in memory (nothing exported yet).
End:   a fresh Base+ session is ready to process s+1 (after SAM2's propagate preflight).

  resume / paper / conv_strip_vm : export the raw state to canonical form (+ translate) + inject + preflight
  restart_p0 / restart_mask / restart_p0_mask : read the prompt JPEGs from disk, decode, resize, add_new_mask each,
                                                preflight (each stage also timed on its own)

The rollout to s is not timed. One warm-up, then REPEAT timed runs; the first (cold) call is reported too.

  . env.sh && CUDA_VISIBLE_DEVICES=0 python3 code/cost_single.py --method paper --cases all --out checks/all/paper.json

--cases all: the 188 switches of the accuracy run (PUMaVOS 78, M3-VOS 110), read from runs/unseen/*/cases.
--cases six: the earlier six-case check (checks/single). One rollout per case serves all of its switches.
"""
import argparse
import glob
import json
import time

import numpy as np
import torch

import large_vos_eval  # noqa: F401  paper / strip: sources
import unseen_eval as live
from jf_eval import Sources, fresh
from vos_memory_inspector.sam2_state import canonicalize_sam2_inference_state, inject_sam2_canonical_state

CASES = [('pumavos', 'billie_hair'), ('pumavos', 'caps'), ('pumavos', 'chair'),
         ('m3vos', '0001_open_cup_1'), ('m3vos', '0004_open_cup_6'), ('m3vos', '0007_use_measure_1')]
TOKENS = dict(paper='paper', conv_strip_vm='strip:lin:S2B:conv3_mean:pointer_mean:shared:vm')
REPEAT = 10


def all_cases():
    """Every case of the accuracy run (runs/unseen/*/cases): PUMaVOS 78 switches (25/50/75%), M3-VOS 110 (50%)."""
    out = []
    for dataset in ('pumavos', 'm3vos'):
        for f in sorted(glob.glob(f'runs/unseen/{dataset}/cases/*.json')):
            r = json.load(open(f))
            out.append((dataset, r['video'], r['object'], r['start'], r['end'],
                        sorted((v['switch'], k) for k, v in r['switches'].items())))
    return out


def sync_time(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0), out


def track_through(predictor, video, obj, start, switches):
    """Yield (switch, raw inference state, prompt, last visible (frame, mask)) at each switch frame, in order.

    One rollout serves every switch of a case; the state is yielded before anything past the switch is processed."""
    state = fresh(predictor, live.Frames(video, predictor.image_size))
    prompt = video.labels(start)[0] == obj
    last = (start, prompt)
    todo = sorted(switches)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        predictor.add_new_mask(state, frame_idx=start, obj_id=1, mask=torch.as_tensor(prompt, dtype=torch.bool))
        gen = predictor.propagate_in_video(state, start_frame_idx=start, max_frame_num_to_track=todo[-1] - start)
        for frame, _, logits in gen:
            mask = (logits[0, 0] > 0).cpu().numpy()
            entry = state['output_dict_per_obj'][0]['non_cond_frame_outputs'].get(frame) or \
                state['output_dict_per_obj'][0]['cond_frame_outputs'].get(frame)
            if entry is not None and float(entry['object_score_logits'].max()) > 0 and mask.any():
                last = (frame, mask)
            if frame == todo[0]:
                todo.pop(0)
                yield frame, state, prompt, last
                if not todo:
                    break
            # as unseen_eval.rollout: records older than frame - 17 are never read again, so the tracker drops them
            store = state['output_dict_per_obj'][0]['non_cond_frame_outputs']
            for old in [f for f in store if f < frame - 17]:
                del store[old]
        gen.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['resume', 'paper', 'conv_strip_vm', 'restart_p0', 'restart_mask', 'restart_p0_mask'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--cases', choices=['all', 'six'], default='all')
    args = ap.parse_args()
    from sam2.build_sam import build_sam2_video_predictor
    base = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml', str(live.CHECKPOINTS / 'sam2.1_hiera_base_plus.pt'), device='cuda')
    small = None
    if args.method in TOKENS:
        small = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml', str(live.CHECKPOINTS / 'sam2.1_hiera_small.pt'), device='cuda')
    sources = Sources(torch.device('cuda'))
    rows = []
    cases = all_cases()
    if args.cases == 'six':   # the earlier six-case check: first object, 50% switch only
        cases = [c[:5] + ([s for s in c[5] if s[1] == '50'],) for c in cases if (c[0], c[1]) in CASES and c[2] == 1]
    for n, (dataset, name, obj, start, end, switches) in enumerate(cases):
        video = live.Video(dataset, name)
        tracker = small if args.method in TOKENS else base
        labels = dict(switches)
        for switch, raw, prompt, last in track_through(tracker, video, obj, start, list(labels)):
            stages = {}

            if args.method in ('resume',) + tuple(TOKENS):
                def job():
                    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                        t, canon = sync_time(lambda: live.active_state(canonicalize_sam2_inference_state(raw, switch_frame=switch, strict=True)))
                        stages.setdefault('export', []).append(t)
                        if args.method != 'resume':
                            def translate():
                                a, b = sources.resolve(TOKENS[args.method], canon, canon)
                                return canon.with_continuous(spatial_memory=a[None, None].bfloat16().cpu(), object_pointer=b[None, None].float().cpu(),
                                                             presence_logits=canon.presence_logits.clone(),
                                                             positional_information={'policy': 'regenerate_at_target'})
                            t, canon = sync_time(translate)
                            stages.setdefault('translate', []).append(t)
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        session = fresh(base, live.Frames(video, base.image_size))
                        t, _ = sync_time(lambda: inject_sam2_canonical_state(canon, predictor=base, inference_state=session))
                        stages.setdefault('inject', []).append(t)
                        t, _ = sync_time(lambda: base.propagate_in_video_preflight(session))
                        stages.setdefault('preflight', []).append(t)
            else:
                first = (start, prompt)
                prompts = {'restart_p0': [first], 'restart_mask': [last],
                           'restart_p0_mask': [first] + ([last] if last[0] != start else [])}[args.method]

                def job():
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        frames = live.Frames(video, base.image_size)          # empty cache: JPEG read + decode + resize happen below
                        session = fresh(base, frames)
                        for i, (frame, mask) in enumerate(prompts):
                            t, _ = sync_time(lambda: frames[frame])
                            stages.setdefault(f'decode_{i}', []).append(t)
                            t, _ = sync_time(lambda: base.add_new_mask(session, frame_idx=frame, obj_id=1, mask=torch.as_tensor(mask, dtype=torch.bool)))
                            stages.setdefault(f'add_new_mask_{i}', []).append(t)
                        t, _ = sync_time(lambda: base.propagate_in_video_preflight(session))
                        stages.setdefault('preflight', []).append(t)

            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            cold, _ = sync_time(job)
            times = []
            for _ in range(REPEAT):
                t, _ = sync_time(job)
                times.append(t)
            peak = (torch.cuda.max_memory_allocated() - before) / 2 ** 20
            row = dict(dataset=dataset, video=name, object=obj, fraction=labels[switch], switch=switch, cold_ms=cold, median_ms=float(np.median(times)),
                       min_ms=float(np.min(times)), max_ms=float(np.max(times)), peak_mb=peak,
                       stages_median_ms={k: float(np.median(v[1:])) for k, v in stages.items()}, prompt_frames=[f for f, _ in prompts] if args.method.startswith('restart') else [])
            rows.append(row)
            print(args.method, n, dataset, name, obj, labels[switch], f"cold {cold:.2f} ms, median {row['median_ms']:.2f} ms [{row['min_ms']:.2f}, {row['max_ms']:.2f}], peak {peak:.0f} MB",
                  {k: round(v, 2) for k, v in row['stages_median_ms'].items()}, flush=True)
            torch.cuda.empty_cache()
        del raw
        json.dump(dict(method=args.method, gpu=torch.cuda.get_device_name(0), repeat=REPEAT, cases=rows), open(args.out, 'w'), indent=1)
    json.dump(dict(method=args.method, gpu=torch.cuda.get_device_name(0), repeat=REPEAT, cases=rows), open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
