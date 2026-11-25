#!/usr/bin/env python3
"""Is the conditioning record a pure function of (prompt frame, prompt mask)?

Compare the record stored after a full rollout with one made in a fresh session that encodes
only the prompt frame. Exact agreement means one Base+ encode can replace its translation.
"""
import time

import torch

from unseen_eval import CHECKPOINTS, Frames, Video, active_state, fresh
from vos_memory_inspector.sam2_state import canonicalize_sam2_inference_state


def cond_memory(state):
    store = state['output_dict_per_obj'][0]['cond_frame_outputs']
    frame = sorted(store)[0]
    return frame, store[frame]['maskmem_features'].float().cpu(), store[frame]['obj_ptr'].float().cpu()


def main():
    from sam2.build_sam import build_sam2_video_predictor
    base = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml', str(CHECKPOINTS / 'sam2.1_hiera_base_plus.pt'), device='cuda')
    for dataset, name in (('m3vos', '0051_shake_wine_1'), ('pumavos', 'billie_hair')):
        video = Video(dataset, name)
        obj, start, end = video.objects()[0]
        prompt = video.labels(start)[0] == obj
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            # Full rollout for 40 frames.
            state = fresh(base, Frames(video, base.image_size))
            base.add_new_mask(state, frame_idx=start, obj_id=1, mask=torch.as_tensor(prompt, dtype=torch.bool))
            for frame, _, _ in base.propagate_in_video(state, start_frame_idx=start, max_frame_num_to_track=40):
                pass
            f1, a1, p1 = cond_memory(state)
            # Fresh session: only the prompt frame.
            second = fresh(base, Frames(video, base.image_size))
            torch.cuda.synchronize()
            started = time.monotonic()
            base.add_new_mask(second, frame_idx=start, obj_id=1, mask=torch.as_tensor(prompt, dtype=torch.bool))
            base.propagate_in_video_preflight(second)
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            f2, a2, p2 = cond_memory(second)
        print(name, 'frame', f1, f2, 'spatial equal', torch.equal(a1, a2), 'max diff', float((a1 - a2).abs().max()),
              'pointer equal', torch.equal(p1, p2), 'max diff', float((p1 - p2).abs().max()), f'encode {seconds * 1000:.0f} ms', flush=True)


if __name__ == '__main__':
    main()
