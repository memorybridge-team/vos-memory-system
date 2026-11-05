#!/usr/bin/env python3
"""Find what grows on the GPU during a long continuation."""
import gc
import sys
from collections import Counter

import torch

from unseen_eval import CHECKPOINTS, Frames, Video, fresh


def live():
    counts = Counter()
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.is_cuda:
                counts[tuple(o.shape)] += 1
        except Exception:
            pass
    return counts


def main():
    from sam2.build_sam import build_sam2_video_predictor
    mode = sys.argv[1]
    overrides = ['++model.fill_hole_area=0'] if mode == 'nofill' else []
    base = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml', str(CHECKPOINTS / 'sam2.1_hiera_base_plus.pt'),
                                      device='cuda', hydra_overrides_extra=overrides)
    print('fill_hole_area', base.fill_hole_area, flush=True)
    video = Video('pumavos', 'billie_hair')
    obj, start, end = video.objects()[0]
    prompt = video.labels(start)[0] == obj
    frames = Frames(video, base.image_size)
    state = fresh(base, frames)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        base.add_new_mask(state, frame_idx=start, obj_id=1, mask=torch.as_tensor(prompt, dtype=torch.bool))
        previous = None
        for frame, ids, logits in base.propagate_in_video(state, start_frame_idx=start, max_frame_num_to_track=int(sys.argv[2])):
            store = state['output_dict_per_obj'][0]['non_cond_frame_outputs']
            for old in [f for f in store if f < frame - 17]:
                del store[old]
            if mode == 'clear':
                torch.clear_autocast_cache()
            if frame % 50 == 0:
                allocated = torch.cuda.memory_allocated() / 2 ** 20
                print(f'frame {frame} allocated {allocated:.0f} MiB reserved {torch.cuda.memory_reserved() / 2 ** 20:.0f} MiB python_tensors {sum(live().values())}', flush=True)
                if frame == 100:
                    found = [o for o in gc.get_objects() if torch.is_tensor(o) and o.is_cuda and tuple(o.shape) == (1, 1, 256, 256)]
                    print('tensors', len(found), 'dtypes', Counter(str(t.dtype) for t in found), flush=True)
                    for t in found[:1] + found[len(found) // 2:len(found) // 2 + 1] + found[-1:]:
                        for r in gc.get_referrers(t):
                            if r is found or r is locals() or isinstance(r, type(sys._getframe())):
                                continue
                            kind = type(r).__name__
                            detail = list(r.keys())[:8] if isinstance(r, dict) else (len(r) if isinstance(r, (list, tuple)) else repr(r)[:100])
                            print('  referrer', kind, detail, flush=True)
                            for q in gc.get_referrers(r)[:4]:
                                if q is found or isinstance(q, type(sys._getframe())):
                                    continue
                                d = list(q.keys())[:6] if isinstance(q, dict) else repr(q)[:120]
                                print('      held by', type(q).__name__, d, flush=True)
                if frame in (100, 200):
                    current = live()
                    if previous is not None:
                        print('  grew:', (current - previous).most_common(6), flush=True)
                    previous = current


if __name__ == '__main__':
    main()
