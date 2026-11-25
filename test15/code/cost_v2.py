#!/usr/bin/env python3
"""Switch cost, protocol v2 (DESIGN-cost-v2.md). One method per process, alone on GPU 0.

Interval (same for every method)
  start: the tracker that ran up to s holds its raw SAM2 inference state in memory; nothing is exported yet.
  end:   a fresh Base+ session is ready to process s+1 (after SAM2's propagate preflight).

  resume / paper / conv_strip_vm         : export -> (translate) -> inject -> preflight
  restart_* / olv / lastvis              : per prompt: read+decode+resize its JPEG, add_new_mask; then preflight

Per switch: 1 cold call (kept apart), WARMUP discarded calls, REPEAT timed calls (all raw times are saved).
After timing, one more call builds a session that is propagated to s+1; its predicted area is the equivalence
fingerprint against the accuracy run (gate G2). The rollout to s is not timed, but each tracker step is timed on
its own (per-frame latency, for context).

  . env.sh && CUDA_VISIBLE_DEVICES=0 python3 code/cost_v2.py --method paper --out checks/v2/paper.json [--smoke]
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import large_vos_eval  # noqa: F401  paper / strip: sources
import unseen_eval as live
from jf_eval import Sources, fresh
from vos_memory_inspector.sam2_state import canonicalize_sam2_inference_state, inject_sam2_canonical_state

TOKENS = dict(paper='paper', conv_strip_vm='strip:lin:S2B:conv3_mean:pointer_mean:shared:vm')
STATE_METHODS = ('resume',) + tuple(TOKENS)
PROMPT_METHODS = ('restart_p0', 'restart_mask', 'restart_p0_mask', 'olv', 'lastvis')
SMALL_TRACKER = tuple(TOKENS) + ('olv', 'lastvis')     # these continue from Small's run; the rest from Base+'s
FRACTIONS = dict(pumavos=(0.25, 0.5, 0.75), m3vos=(0.5,))
WARMUP, REPEAT = 3, 20
FIELDS = ('spatial_memory', 'object_pointer', 'frame_indices', 'slot_order', 'is_conditioning', 'validity')
SMOKE = [('pumavos', 'billie_hair'), ('pumavos', 'caps'), ('m3vos', '0001_open_cup_1'), ('m3vos', '0004_open_cup_6')]
ACCURACY = Path('runs/unseen')


def video_names(dataset):
    if dataset == 'm3vos':
        return [x.strip() for x in (live.DATA / 'M3VOS-manual/data/ImageSets/val.txt').read_text().splitlines() if x.strip()]
    return sorted(p.name for p in (live.DATA / 'PUMaVOS/extracted/PUBLIC_PUMaVOS/JPEGImages').iterdir() if p.is_dir())


def switch_of(start, end, fraction):
    return min(max(start + round(fraction * (end - start)), start + 1), end - 1)   # as resume_eval / unseen_eval


def sync_time(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0), out


def gpu_snapshot():
    """Other compute processes on physical GPU 0, and its SM clock / temperature / power right now."""
    apps = subprocess.run(['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.split()
    others = [int(p) for p in apps if p.strip().isdigit() and int(p) != os.getpid()]
    q = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=clocks.sm,temperature.gpu,power.draw', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True).stdout.strip().split(',')
    return dict(other_pids=others, sm_mhz=float(q[0]), temp_c=float(q[1]), power_w=float(q[2]))


def sentinel(base):
    """Fixed GPU workload (Base+ image encoder on a fixed input), median of 20; detects drift within a process."""
    x = torch.randn(1, 3, base.image_size, base.image_size, device='cuda', generator=torch.Generator('cuda').manual_seed(0))
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(3):
            base.forward_image(x)
        return float(np.median([sync_time(lambda: base.forward_image(x))[0] for _ in range(20)]))


def track_through(predictor, video, obj, start, switches, steps):
    """Yield (switch, raw state, prompt, last visible (frame, mask)) at each switch, in order (as unseen_eval.rollout)."""
    state = fresh(predictor, live.Frames(video, predictor.image_size))
    prompt = video.labels(start)[0] == obj
    last = (start, prompt)
    todo = sorted(switches)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        predictor.add_new_mask(state, frame_idx=start, obj_id=1, mask=torch.as_tensor(prompt, dtype=torch.bool))
        gen = predictor.propagate_in_video(state, start_frame_idx=start, max_frame_num_to_track=todo[-1] - start)
        try:
            while True:
                t, item = sync_time(lambda: next(gen, None))
                if item is None:
                    break
                frame, _, logits = item
                if frame > start:
                    steps.append(t)
                mask = (logits[0, 0] > 0).cpu().numpy()
                memory = state['output_dict_per_obj'][0]
                entry = memory['non_cond_frame_outputs'].get(frame) or memory['cond_frame_outputs'].get(frame)
                if entry is not None and float(entry['object_score_logits'].max()) > 0 and mask.any():
                    last = (frame, mask)
                if frame == todo[0]:
                    todo.pop(0)
                    yield frame, state, prompt, last
                    if not todo:
                        break
                store = memory['non_cond_frame_outputs']
                for old in [f for f in store if f < frame - 17]:
                    del store[old]
        finally:
            gen.close()


def accuracy_area(dataset, name, obj, label, method):
    """Predicted area at s+1 in the accuracy run (case file), or None when that case was not run there."""
    path = ACCURACY / dataset / 'cases' / f'{name}_{obj}.json'
    if not path.exists():
        return None
    row = json.loads(path.read_text())['switches'].get(label)
    if row is None or method not in row.get('trace', {}):
        return None
    return row['trace'][method]['area'][0]


def save(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=STATE_METHODS + PROMPT_METHODS)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true', help='four cases only (gate check before the full run)')
    args = ap.parse_args()
    out = Path(args.out)
    from sam2.build_sam import build_sam2_video_predictor
    base = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml', str(live.CHECKPOINTS / 'sam2.1_hiera_base_plus.pt'), device='cuda')
    small = None
    if args.method in SMALL_TRACKER:
        small = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml', str(live.CHECKPOINTS / 'sam2.1_hiera_small.pt'), device='cuda')
    tracker = small if args.method in SMALL_TRACKER else base
    sources = Sources(torch.device('cuda'))

    payload = json.loads(out.read_text()) if out.exists() else dict(method=args.method, rows=[], sentinels=[], processes=[])
    done = {(r['dataset'], r['video'], r['object'], r['fraction']) for r in payload['rows']}
    payload['processes'].append(dict(pid=os.getpid(), started=time.strftime('%Y-%m-%d %H:%M:%S'), gpu=torch.cuda.get_device_name(0),
                                     torch=torch.__version__, cuda=torch.version.cuda, warmup=WARMUP, repeat=REPEAT, first_row=len(payload['rows'])))
    payload['sentinels'].append(dict(at=len(payload['rows']), when='start', ms=sentinel(base)))
    save(out, payload)

    cases = [(d, n) for d in ('pumavos', 'm3vos') for n in video_names(d)]
    if args.smoke:
        cases = [c for c in cases if c in SMOKE]
    first_in_process = True
    for dataset, name in cases:
        video = live.Video(dataset, name)
        for obj, start, end in video.objects():
            labels = {switch_of(start, end, f): str(round(f * 100)) for f in FRACTIONS[dataset]}
            if all((dataset, name, obj, lab) in done for lab in labels.values()):
                continue
            steps = []
            for switch, raw, prompt, last in track_through(tracker, video, obj, start, list(labels), steps):
                label = labels[switch]
                if (dataset, name, obj, label) in done:
                    continue
                stages, session_box = {}, {}
                if args.method in STATE_METHODS:
                    def job():
                        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                            t, canon = sync_time(lambda: live.active_state(canonicalize_sam2_inference_state(raw, switch_frame=switch, strict=True)))
                            stages.setdefault('export', []).append(t)
                            if args.method in TOKENS:
                                def translate():
                                    a, b = sources.resolve(TOKENS[args.method], canon, canon)
                                    return canon.with_continuous(spatial_memory=a[None, None].bfloat16().cpu(), object_pointer=b[None, None].float().cpu(),
                                                                 presence_logits=canon.presence_logits.clone(),
                                                                 positional_information={'policy': 'regenerate_at_target'})
                                t, canon = sync_time(translate)
                                stages.setdefault('translate', []).append(t)
                        session_box['canon'] = canon
                        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                            session = fresh(base, live.Frames(video, base.image_size))
                            t, _ = sync_time(lambda: inject_sam2_canonical_state(canon, predictor=base, inference_state=session))
                            stages.setdefault('inject', []).append(t)
                            t, _ = sync_time(lambda: base.propagate_in_video_preflight(session))
                            stages.setdefault('preflight', []).append(t)
                        session_box['session'] = session
                    prompt_frames = []
                else:
                    first = (start, prompt)
                    # Base+ tracker: last = Base+'s last visible (resume_eval lv_b); Small tracker: Small's (lv_s)
                    prompts = {'restart_p0': [first], 'restart_mask': [last], 'lastvis': [last],
                               'restart_p0_mask': [first] + ([last] if last[0] != start else []),
                               'olv': [first] + ([last] if last[0] != start else [])}[args.method]
                    prompt_frames = [f for f, _ in prompts]
                    session_box['bytes'] = sum(os.path.getsize(video.frames[f]) + video.height * video.width // 8 for f in prompt_frames)

                    def job():
                        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                            frames = live.Frames(video, base.image_size)    # empty cache: JPEG read + decode + resize are timed
                            session = fresh(base, frames)
                            for i, (frame, mask) in enumerate(prompts):
                                t, _ = sync_time(lambda: frames[frame])
                                stages.setdefault(f'decode_{i}', []).append(t)
                                t, _ = sync_time(lambda: base.add_new_mask(session, frame_idx=frame, obj_id=1, mask=torch.as_tensor(mask, dtype=torch.bool)))
                                stages.setdefault(f'add_new_mask_{i}', []).append(t)
                            t, _ = sync_time(lambda: base.propagate_in_video_preflight(session))
                            stages.setdefault('preflight', []).append(t)
                        session_box['session'] = session

                gpu_before = gpu_snapshot()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                alloc0, reserved0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
                cold, _ = sync_time(job)
                for _ in range(WARMUP):
                    sync_time(job)
                times = [sync_time(job)[0] for _ in range(REPEAT)]
                peak_alloc = (torch.cuda.max_memory_allocated() - alloc0) / 2 ** 20
                peak_reserved = (torch.cuda.max_memory_reserved() - reserved0) / 2 ** 20
                gpu_after = gpu_snapshot()
                n_untimed = 1 + WARMUP
                stage_ms = {k: v[n_untimed:] for k, v in stages.items()}

                # G2 fingerprint: one more session, propagated to s+1 (not timed)
                sync_time(job)
                session = session_box['session']
                if 'canon' in session_box:   # handoff size, counted outside the timed calls
                    canon = session_box['canon']
                    session_box['bytes'] = 16 + sum(int(getattr(canon, n)[0, 0][canon.validity[0, 0]].numel() * getattr(canon, n).element_size()) for n in FIELDS)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    gen = base.propagate_in_video(session, start_frame_idx=switch + 1, max_frame_num_to_track=0)
                    frame, _, logits = next(gen)
                    gen.close()
                pixels = int((logits[0, 0] > 0).sum())
                area = pixels / logits[0, 0].numel()
                ref = accuracy_area(dataset, name, obj, label, args.method)

                row = dict(dataset=dataset, video=name, object=obj, fraction=label, start=start, end=end, switch=switch,
                           first_in_process=first_in_process, cold_ms=cold, times_ms=times,
                           median_ms=float(np.median(times)), stages_ms=stage_ms,
                           peak_alloc_mb=peak_alloc, peak_reserved_mb=peak_reserved, handoff_bytes=session_box['bytes'],
                           prompt_frames=prompt_frames, area_s1=area, pixels_s1=pixels, area_s1_accuracy_run=ref,
                           pixels_s1_accuracy_run=None if ref is None else int(round(ref * logits[0, 0].numel())), s1_frame=int(frame),
                           gpu_before=gpu_before, gpu_after=gpu_after)
                payload['rows'].append(row)
                done.add((dataset, name, obj, label))
                first_in_process = False
                print(args.method, len(payload['rows']), dataset, name, obj, label,
                      f"cold {cold:.2f} median {row['median_ms']:.2f} ms peak {peak_alloc:.0f} MB area {area:.5f} ref {ref}",
                      'OTHER_GPU_PROCESS' if gpu_before['other_pids'] or gpu_after['other_pids'] else '', flush=True)
                del session_box
                torch.cuda.empty_cache()
            payload.setdefault('tracker_steps', []).append(dict(dataset=dataset, video=name, object=obj,
                                                                median_ms=float(np.median(steps[10:] or steps)), n=len(steps)))
            if len(payload['rows']) // 25 != (len(payload['rows']) - len(labels)) // 25:
                payload['sentinels'].append(dict(at=len(payload['rows']), when='periodic', ms=sentinel(base)))
            save(out, payload)
    payload['sentinels'].append(dict(at=len(payload['rows']), when='end', ms=sentinel(base)))
    payload['complete'] = True
    save(out, payload)
    print('COMPLETE', args.method, len(payload['rows']), flush=True)


if __name__ == '__main__':
    main()
