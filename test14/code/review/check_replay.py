#!/usr/bin/env python3
"""Where does a live rollout part from a cached pair? Per-record error for Small and Base+."""
import sys

import numpy as np
import torch

from handoff_eval import LazyFrames, fresh, load_label, CHECKPOINTS
from jf_eval import select_cases, video_paths
from unseen_eval import active_state
from vos_memory_inspector.sam2_state import canonicalize_sam2_inference_state
from vos_memory_inspector.state_schema import CanonicalState, StateSpec
from sam2.build_sam import build_sam2_video_predictor

SCALE_S, SCALE_P = 0.80421, 0.63377


def replay(predictor, paths, start, prompt, switch, autocast):
    state = fresh(predictor, LazyFrames(paths, predictor.image_size))
    exported = None
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
        predictor.add_new_mask(state, frame_idx=start, obj_id=1, mask=torch.as_tensor(prompt, dtype=torch.bool))
        for frame, _, _ in predictor.propagate_in_video(state, start_frame_idx=start, max_frame_num_to_track=switch - start):
            if frame == switch:
                exported = active_state(canonicalize_sam2_inference_state(state, switch_frame=frame, strict=True))
            store = state['output_dict_per_obj'][0]['non_cond_frame_outputs']
            for old in [f for f in store if f < frame - 17]:
                del store[old]
    return exported


def report(name, live, cached):
    x, y = live.spatial_memory[0, 0].float(), cached.spatial_memory[0, 0].float()
    p, q = live.object_pointer[0, 0].float(), cached.object_pointer[0, 0].float()
    sp = (x - y).square().mean((1, 2, 3)) / SCALE_S ** 2
    pt = (p - q).square().mean(1) / SCALE_P ** 2
    print(f'  {name}: spatial nMSE per record {np.round(sp.numpy(), 5).tolist()}')
    print(f'  {" " * len(name)}  pointer nMSE per record {np.round(pt.numpy(), 5).tolist()}')


small = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml', str(CHECKPOINTS / 'sam2.1_hiera_small.pt'), device='cuda')
base = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml', str(CHECKPOINTS / 'sam2.1_hiera_base_plus.pt'), device='cuda')
for row in select_cases('dev', 1, 2, 7, 'MOSEv2'):
    with torch.serialization.safe_globals([CanonicalState, StateSpec]):
        value = torch.load(row['path'], map_location='cpu', weights_only=True)
    s, t = value['source_canonical'], value['target_canonical']
    switch = int(s.switch_frame)
    paths, gt = video_paths(row['dataset'], row['video_id'])
    cond = (s.validity[0, 0] & s.is_conditioning[0, 0]).nonzero().flatten().tolist()
    start = int(s.frame_indices[0, 0, cond[0]])
    prompt, _ = load_label(gt, paths[start], int(row['object_id']))
    print(row['case_id'], 'start', start, 'switch', switch, 'prompt area', round(float(prompt.mean()), 4), 'dtype', s.spatial_memory.dtype)
    print('  metadata', {k: v for k, v in s.metadata.items() if k in ('source', 'storage_device', 'compute_device')})
    for autocast in (True, False):
        report(f'small autocast={autocast}', replay(small, paths, start, prompt, switch, autocast), s)
        report(f'base  autocast={autocast}', replay(base, paths, start, prompt, switch, autocast), t)
