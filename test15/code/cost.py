#!/usr/bin/env python3
"""Switch cost of each memory-resume-20261120 method, measured as in cmmt-design-review-20261104/code/cost_eval.py.

Time runs from Base+ taking over at s until it is ready to process s+1 (state injection or prompt encoding, then
SAM2's propagate preflight). Median of 3 repeats after one warm-up; peak GPU memory above what was allocated before.
Frames are preloaded in RAM, so disk reads do not count. Bytes: the exported state for resume and the translators;
the RGB files re-encoded and the binary masks (1 bit per pixel) for the restarts.

  . env.sh && CUDA_VISIBLE_DEVICES=0 python3 code/cost.py --dataset pumavos --max-cases 10 --out checks/cost_pumavos.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import large_vos_eval  # noqa: F401  paper / strip: sources
import unseen_eval as live
from jf_eval import Sources, fresh
from vos_memory_inspector.sam2_state import inject_sam2_canonical_state
from vos_memory_inspector.vos_metrics import load_official_metrics

FIELDS = ('spatial_memory', 'object_pointer', 'frame_indices', 'slot_order', 'is_conditioning', 'validity')
TRANSLATORS = dict(paper='paper', conv_strip_vm='strip:lin:S2B:conv3_mean:pointer_mean:shared:vm')


class Preloaded:
    def __init__(self, video, size, wanted):
        lazy = live.Frames(video, size)
        self.count, self.height, self.width = len(lazy), lazy.height, lazy.width
        self.frames = {i: lazy[i].clone() for i in sorted(set(wanted))}

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return self.frames[index]


def timed(job, repeat=3):
    job()
    times, peaks = [], []
    for _ in range(repeat):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        started = time.perf_counter()
        job()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        peaks.append((torch.cuda.max_memory_allocated() - before) / 2 ** 20)
    return float(np.median(times)), float(np.max(peaks))


def payload(state):
    keep = state.validity[0, 0]
    return 16 + sum(int(getattr(state, n)[0, 0][keep].numel() * getattr(state, n).element_size()) for n in FIELDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--fraction', type=float, default=0.5)
    ap.add_argument('--max-cases', type=int, default=10)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    metrics, reason = load_official_metrics()
    if metrics is None:
        raise RuntimeError(reason)
    from sam2.build_sam import build_sam2_video_predictor
    small = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml', str(live.CHECKPOINTS / 'sam2.1_hiera_small.pt'), device='cuda')
    base = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml', str(live.CHECKPOINTS / 'sam2.1_hiera_base_plus.pt'), device='cuda')
    sources = Sources(torch.device('cuda'))
    if args.dataset == 'm3vos':
        names = [x.strip() for x in (live.DATA / 'M3VOS-manual/data/ImageSets/val.txt').read_text().splitlines() if x.strip()][:81]
    else:
        names = sorted(p.name for p in (live.DATA / 'PUMaVOS/extracted/PUBLIC_PUMaVOS/JPEGImages').iterdir() if p.is_dir())
    rows = []
    for name in names:
        if len(rows) >= args.max_cases:
            break
        video = live.Video(args.dataset, name)
        obj, start, end = video.objects()[0]
        switch = min(max(start + round(args.fraction * (end - start)), start + 1), end - 1)
        _, sources_at, small_anchor = live.rollout(small, video, obj, start, switch + 1, {switch}, metrics)
        _, targets_at, base_anchor = live.rollout(base, video, obj, start, switch + 1, {switch}, metrics)
        s, t = sources_at[switch], targets_at[switch]
        lv_b = base_anchor[switch]
        prompt = torch.as_tensor(video.labels(start)[0] == obj, dtype=torch.bool)
        frames = Preloaded(video, base.image_size, [start, lv_b[0]])
        mask_bytes = video.height * video.width // 8

        def restart(prompts):
            def job():
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    state = fresh(base, frames)
                    for frame, mask in prompts:
                        base.add_new_mask(state, frame_idx=frame, obj_id=1, mask=torch.as_tensor(mask, dtype=torch.bool))
                    base.propagate_in_video_preflight(state)
            return job

        def inject(token):
            def job():
                with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                    if token is None:
                        state = t
                    else:
                        a, b = sources.resolve(token, s, t)
                        state = s.with_continuous(spatial_memory=a[None, None].bfloat16().cpu(), object_pointer=b[None, None].float().cpu(),
                                                  presence_logits=s.presence_logits.clone(), positional_information={'policy': 'regenerate_at_target'})
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    session = fresh(base, frames)
                    inject_sam2_canonical_state(state, predictor=base, inference_state=session)
                    base.propagate_in_video_preflight(session)
            return job

        p0 = [(start, prompt)]
        mine = [lv_b] if lv_b[0] != start else []
        jobs = dict(resume=inject(None), restart_p0=restart(p0), restart_mask=restart([lv_b]), restart_p0_mask=restart(p0 + mine))
        jobs.update({m: inject(token) for m, token in TRANSLATORS.items()})
        row = dict(video=name, object=obj, switch=switch, seconds={}, peak_mb={}, state_bytes=payload(t),
                   bytes=dict(restart_p0=os.path.getsize(video.frames[start]) + mask_bytes,
                              restart_mask=os.path.getsize(video.frames[lv_b[0]]) + mask_bytes,
                              restart_p0_mask=os.path.getsize(video.frames[start]) + mask_bytes
                              + (os.path.getsize(video.frames[lv_b[0]]) + mask_bytes if mine else 0)))
        for m, job in jobs.items():
            row['seconds'][m], row['peak_mb'][m] = timed(job)
        rows.append(row)
        print(name, {m: round(1000 * v, 2) for m, v in row['seconds'].items()}, 'ms', flush=True)
    summary = {m: dict(ms_median=1000 * float(np.median([r['seconds'][m] for r in rows])),
                       peak_mb_max=float(np.max([r['peak_mb'][m] for r in rows]))) for m in rows[0]['seconds']}
    summary['state_bytes_median'] = float(np.median([r['state_bytes'] for r in rows]))
    for m in ('restart_p0', 'restart_mask', 'restart_p0_mask'):
        summary[m]['bytes_median'] = float(np.median([r['bytes'][m] for r in rows]))
    with open(args.out, 'w') as f:
        json.dump(dict(dataset=args.dataset, fraction=args.fraction, gpu=torch.cuda.get_device_name(0), summary=summary, cases=rows), f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
