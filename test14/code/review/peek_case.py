#!/usr/bin/env python3
"""Draws what each method tracks in one cached case: predicted masks over the frames, GT as a green outline."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import handoff_eval as he
from handoff_eval import (CHECKPOINTS, CanonicalState, Sources, StateSpec, build, load_label, load_official_metrics,
                          parse_methods, run_methods, select_cases, small_track, video_paths)

METHODS = ['target_ref=target', 'orig=orig', 'lastvis=lastvis', 'olv=olv', 'ridge_nc=target,ridge,target,ridge',
           'direct_nc=target,direct,target,direct', 'ridge_nc_anchor=anchorc:ridge']
WIDTH = 230


def box(gt, width, height):
    if gt is None or not gt.any():
        return None
    ys, xs = np.nonzero(gt)
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * 4
    side = int(min(max(side, 160), min(width, height)))
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    left = int(np.clip(cx - side // 2, 0, width - side))
    top = int(np.clip(cy - side // 2, 0, height - side))
    return left, top, left + side, top + side


def overlay(path, mask, gt, crop=False):
    image = Image.open(path).convert('RGB')
    w, h = image.size
    pixels = np.asarray(image).astype(np.float32)
    if mask is not None and mask.any():
        pixels[mask] = pixels[mask] * .45 + np.array([255, 40, 40]) * .55
    canvas = Image.fromarray(pixels.astype(np.uint8))
    if gt is not None and gt.any():
        edge = gt & ~(np.roll(gt, 2, 0) & np.roll(gt, -2, 0) & np.roll(gt, 2, 1) & np.roll(gt, -2, 1))
        drawn = np.asarray(canvas).copy()
        drawn[edge] = (40, 255, 40)
        canvas = Image.fromarray(drawn)
    region = box(gt, w, h) if crop else None
    if crop:
        return (canvas.crop(region) if region else canvas.crop((0, 0, min(w, h), min(w, h)))).resize((WIDTH, WIDTH))
    return canvas.resize((WIDTH, int(h * WIDTH / w)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='val')
    parser.add_argument('--dataset')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--video', required=True)
    parser.add_argument('--offsets', default='1,2,3,5,10,20,50')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--out', default='/workspace/cmmt-design-review-20261104/runs/peek')
    args = parser.parse_args()
    he.AUTOCAST[0] = True
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('REVIEW_GPU_FRACTION', 0.3)))
    offsets = [int(x) for x in args.offsets.split(',')]
    metrics, _ = load_official_metrics()
    from sam2.build_sam import build_sam2_video_predictor
    small = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml', str(CHECKPOINTS / 'sam2.1_hiera_small.pt'), device='cuda')
    base = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml', str(CHECKPOINTS / 'sam2.1_hiera_base_plus.pt'), device='cuda')
    sources = Sources(torch.device('cuda'))
    cases = select_cases(args.split, 1, args.limit, 7, args.dataset)
    row = next(c for c in cases if c['video_id'] == args.video)
    with torch.serialization.safe_globals([CanonicalState, StateSpec]):
        value = torch.load(row['path'], map_location='cpu', weights_only=True)
    s, t = value['source_canonical'], value['target_canonical']
    switch = int(s.switch_frame)
    paths, gt = video_paths(row['dataset'], row['video_id'])
    end = min(len(paths) - 1, switch + max(offsets))
    offsets = [o for o in offsets if switch + o <= end]
    object_id = int(row['object_id'])
    cond = (s.validity[0, 0] & s.is_conditioning[0, 0]).nonzero().flatten().tolist()
    start = int(s.frame_indices[0, 0, cond[0]])
    prompt, _ = load_label(gt, paths[start], object_id)

    # Keep each method's mask at the frames we draw.
    original = he.frame_scores
    kept = {}

    def capture(store, gt_dir, path, oid, mask, table):
        original(store, gt_dir, path, oid, mask, table)
        kept.setdefault(id(store), []).append(mask.copy() if len(store['J']) in offsets else None)

    he.frame_scores = capture
    small_row, last, _ = small_track(small, paths, start, prompt, switch, end, object_id, gt, metrics, False)
    methods = parse_methods(METHODS)
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        built = [(m, *build(m, spec, sources, s, t, last, (start, prompt))) for m, spec in methods.items()]
    scores, _ = run_methods(base, paths, built, switch, object_id, end, gt, metrics)
    scores['small'] = small_row
    print(f"{row['case_id']} prompt frame {start} switch {switch} last visible {last[0]} frames {len(paths)} records {int(s.validity.sum())}")
    order = ['small', 'target_ref', 'orig', 'lastvis', 'olv', 'ridge_nc', 'ridge_nc_anchor', 'direct_nc']
    print('offset  gt%   ' + ' '.join(f'{m[:9]:>9s}' for m in order))
    for index in range(len(scores['small']['J'])):
        if scores['small']['J'][index] is None or index > 40:
            continue
        cells = ' '.join(f"{scores[m]['J'][index]:9.2f}" for m in order)
        print(f"{index + 1:6d} {'vis' if scores['small']['visible'][index] else 'abs'}    {cells}")
    print('mean area % over the drawn range: ' + ' '.join(f"{m} {np.mean(scores[m]['area']) * 100:.2f}" for m in order))

    height = WIDTH if args.crop else int(Image.open(paths[start]).height * WIDTH / Image.open(paths[start]).width)
    columns = ['prompt', 'last visible'] + [f'+{o}' for o in offsets]
    sheet = Image.new('RGB', (110 + WIDTH * len(columns), 22 + (height + 4) * len(order)), 'white')
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(columns):
        draw.text((116 + WIDTH * c, 5), name, fill='black')
    for r, method in enumerate(order):
        y = 22 + (height + 4) * r
        draw.text((4, y + 4), method, fill='black')
        sheet.paste(overlay(paths[start], prompt, prompt, args.crop), (110, y))
        lv, _ = load_label(gt, paths[last[0]], object_id)
        sheet.paste(overlay(paths[last[0]], last[1], lv, args.crop), (110 + WIDTH, y))
        for c, offset in enumerate(offsets):
            target, _ = load_label(gt, paths[switch + offset], object_id)
            sheet.paste(overlay(paths[switch + offset], kept[id(scores[method])][offset - 1], target, args.crop), (110 + WIDTH * (c + 2), y))
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{row['video_id']}{'_crop' if args.crop else ''}.jpg"
    sheet.save(folder / name, quality=80)
    print('saved', folder / name)


if __name__ == '__main__':
    main()
