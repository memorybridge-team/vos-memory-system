#!/usr/bin/env python3
"""Fine-tune a spatial translator against Base+'s downstream output (prototype, fit cases only).

Base+ stays frozen. One sample is a fit case and a frame after its switch. The records go through Base+'s
own memory attention and mask decoder on that frame, once as true Base+ records (teacher) and once with
translated records (student), and the student is pulled toward the teacher's memory-conditioned feature,
mask and object score. Future frames are training data only; at run time the translator uses no past RGB.

Recent records stop at the switch, so for a later frame they are read as if they were adjacent. That keeps
the conditioning record, which stays until the end of a video, under load far from the switch.

Scenarios: `cond` translates the conditioning record and keeps true recents. `all` also translates the records
read at switch+1. `alone` reads the conditioning record with no recents, the state a track is in after an
occlusion: the record must find the object by itself. Once SAM2 calls an object absent its masks become a
constant, so a lost frame has no mask gradient and only the object-score terms can repair it.

--probe skips training and prints the same measures for fixed translators, to see whether they rank
methods the way full-video J&F does.
"""
import argparse
import hashlib
import json
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.nn import functional as F

from jf_eval import LEGACY, ROOT, SCALES, restore_model, video_paths
from review_data import Split
from review_models import make_spatial

CHECKPOINT = '/workspace/j700qtcgrg/CMMT/checkpoints/sam2.1_hiera_base_plus.pt'
OFFSETS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89)  # fixed so held-out sets stay comparable across runs
TRAIN_OFFSETS = OFFSETS + (144, 233)  # the conditioning record stays to the end of long videos
RECENT_READ = 5  # records this close to the switch are read at switch+1
RECORD = 64 * 64 * 64 * 2
POINTER = 256 * 4
MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]


def amp():
    return torch.autocast('cuda', dtype=torch.bfloat16)


class Store:
    """Whole cases from the flat arrays, plus one future frame."""

    def __init__(self, name):
        split = Split(name)
        folder = ROOT / 'data' / name
        self.a = os.open(folder / 'src_sp.i16', os.O_RDONLY)
        self.b = os.open(folder / 'tgt_sp.i16', os.O_RDONLY)
        self.q = os.open(folder / 'tgt_p.f32', os.O_RDONLY)
        self.p = os.open(folder / 'src_p.f32', os.O_RDONLY)
        self.cases = split.meta['cases']
        owner = split.meta['case_id'].numpy()
        starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
        ends = np.r_[starts[1:], len(owner)]
        self.spans = {int(owner[s]): (int(s), int(e)) for s, e in zip(starts, ends)}
        self.frame = split.meta['frame'].numpy().astype(np.int64)
        self.cond = split.meta['cond'].numpy().astype(bool)
        self.lvos = [c['dataset'] != 'MOSEv2' for c in self.cases]
        self.videos = {}

    def paths(self, row):
        key = row['video_id']
        if key not in self.videos:
            self.videos[key] = video_paths(row['dataset'], key)[0]
        return self.videos[key]

    def _read(self, fd, unit, start, count, array):
        os.preadv(fd, [memoryview(array)], start * unit)
        return array

    def item(self, case, offset):
        row = self.cases[case]
        start, end = self.spans[case]
        paths = self.paths(row)
        switch = int(row['runtime_switch'])  # LVOS frame numbers are 5x the runtime index
        offset = min(offset, len(paths) - 1 - switch)
        if offset < 1:
            return None
        count = end - start
        shape = (count, 64, 64, 64)
        x = self._read(self.a, RECORD, start, count, np.empty(shape, dtype=np.int16))
        y = self._read(self.b, RECORD, start, count, np.empty(shape, dtype=np.int16))
        q = self._read(self.q, POINTER, start, count, np.empty((count, 256), dtype=np.float32))
        p = self._read(self.p, POINTER, start, count, np.empty((count, 256), dtype=np.float32))
        from sam2.utils.misc import _load_img_as_tensor
        image, _, _ = _load_img_as_tensor(str(paths[switch + offset]), 1024)
        return dict(case=case, x=torch.from_numpy(x), y=torch.from_numpy(y), q=torch.from_numpy(q), p=torch.from_numpy(p),
                    frames=self.frame[start:end].copy(), cond=self.cond[start:end].copy(), switch=switch,
                    count=len(paths), image=(image.float() - MEAN) / STD, offset=offset)


class Feed:
    """Random training items loaded ahead by threads."""

    def __init__(self, store, seed, workers=4, depth=12, lvos_share=0.):
        self.store, self.pool = store, ThreadPoolExecutor(workers)
        self.rng = np.random.default_rng(seed)
        self.cases = np.array(sorted(store.spans))
        self.lvos_cases = np.array([c for c in self.cases if store.lvos[c]])
        self.lvos_share = lvos_share
        self.pending = deque(self.pool.submit(store.item, *self.draw()) for _ in range(depth))

    def draw(self):
        pool = self.lvos_cases if self.rng.random() < self.lvos_share else self.cases
        return int(self.rng.choice(pool)), int(self.rng.choice(TRAIN_OFFSETS))

    def next(self):
        while True:
            job = self.pending.popleft()
            self.pending.append(self.pool.submit(self.store.item, *self.draw()))
            item = job.result()
            if item is not None:
                return item


def bf16(a, device):
    return a.to(device, non_blocking=True).view(torch.bfloat16)


class Teacher:
    """Frozen Base+ and the two reads that are compared."""

    def __init__(self, device):
        from sam2.build_sam import build_sam2_video_predictor
        self.base = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml', CHECKPOINT, device='cuda')
        self.base.eval().requires_grad_(False)
        self.device = device
        with amp(), torch.no_grad():
            reference = torch.empty((1, 64, 64, 64), device=device, dtype=torch.bfloat16)
            self.position = self.base.memory_encoder.position_encoding(reference).to(torch.bfloat16)[0:1].clone()

    def image(self, image):
        with torch.no_grad(), amp():
            vision = self.base.forward_image(image[None].to(self.device, non_blocking=True))
            _, feats, pos, sizes = self.base._prepare_backbone_features(vision)
        return feats, pos, sizes

    def read(self, view, item, spatial, pointer, alone=False):
        """Base+'s memory-conditioned feature and masks for this image, with these records."""
        feats, pos, sizes = view
        history = dict(cond_frame_outputs={}, non_cond_frame_outputs={})
        for k, frame in enumerate(item['frames']):
            if alone and not item['cond'][k]:
                continue
            key = 'cond_frame_outputs' if item['cond'][k] else 'non_cond_frame_outputs'
            history[key][int(frame)] = dict(maskmem_features=spatial[k][None], maskmem_pos_enc=[self.position], obj_ptr=pointer[k][None])
        base = self.base
        with amp():
            pix = base._prepare_memory_conditioned_features(
                frame_idx=item['switch'] + (item['offset'] if alone else 1), is_init_cond_frame=False, current_vision_feats=feats[-1:], current_vision_pos_embeds=pos[-1:],
                feat_sizes=sizes[-1:], output_dict=history, num_frames=item['count'])
            high = [x.permute(1, 2, 0).view(x.size(1), x.size(2), *s) for x, s in zip(feats[:-1], sizes[:-1])]
            out = base._forward_sam_heads(backbone_features=pix, point_inputs=None, mask_inputs=None, high_res_features=high,
                                          multimask_output=base._use_multimask(False, None))
        return pix.float(), out


def compare(student, teacher):
    pix_s, out_s = student
    pix_t, out_t = teacher
    feature = (pix_s - pix_t).square().mean() / pix_t.var()
    best = out_t[2].argmax(-1)
    mask_s = out_s[0][0, best].clamp(-20, 20)
    mask_t = out_t[0][0, best].clamp(-20, 20)
    mask = F.binary_cross_entropy_with_logits(mask_s, mask_t.sigmoid())
    logit_s, logit_t = out_s[6].float(), out_t[6].float()
    score = F.binary_cross_entropy_with_logits(logit_s, (logit_t > 0).float())
    logit = F.smooth_l1_loss(logit_s, logit_t)
    # BCE is dominated by background on small objects; J counts every object the same.
    p, t = mask_s.sigmoid(), mask_t.sigmoid()
    dice = 1 - 2 * (p * t).sum() / (p.sum() + t.sum() + 1) if (mask_t > 0).any() else p.sum() * 0
    with torch.no_grad():
        a, b = out_s[3] > 0, out_t[3] > 0  # the masks SAM2 would actually pick
        union = (a | b).sum()
        iou = (a & b).sum() / union if union else torch.ones(())
        info = dict(iou=float(iou), gap=float((logit_s - logit_t).mean()), lost=float(((logit_s <= 0) & (logit_t > 0)).float().mean()),
                    extra=float(((logit_s > 0) & (logit_t <= 0)).float().mean()))
    return dict(feature=feature, mask=mask, score=score, dice=dice, logit=logit), info


def used(item):
    return item['cond'] | (item['frames'] >= item['switch'] - RECENT_READ)


def ridge_pointer(device):
    ridge = torch.load(ROOT / 'runs/ridge/ridge.pt', map_location=device, weights_only=True)
    w, b = ridge['pointer_w'], ridge['pointer_b']
    return lambda p: (p.double() @ w + b).float()


def one(teacher, translate, item, scenario, device, pointer=None):
    """Student against teacher on one item. `translate` maps [N,64,64,64] floats to the same shape.

    With `pointer`, translated records also carry translated pointers, as they do at run time."""
    x, y = bf16(item['x'], device).float(), bf16(item['y'], device)
    q = item['q'].to(device, non_blocking=True)
    view = teacher.image(item['image'])
    alone = scenario == 'alone'
    with torch.no_grad():
        target = teacher.read(view, item, y, q, alone)
    keep = torch.from_numpy(used(item) if scenario == 'all' else item['cond']).to(device)
    h = translate(x[keep])
    spatial = y.clone()
    spatial[keep] = h.to(torch.bfloat16)
    if pointer is not None:
        q = q.clone()
        q[keep] = pointer(item['p'].to(device, non_blocking=True)[keep])
    student = teacher.read(view, item, spatial, q, alone)
    terms, info = compare(student, target)
    terms['state'] = (h - y[keep].float()).square().mean() / SCALES['spatial'] ** 2
    return terms, info


def held_out(store, count):
    """Fixed dev items: half LVOS when there are enough, offsets spread evenly."""
    order = sorted(store.spans, key=lambda c: hashlib.sha256(store.cases[c]['case_id'].encode()).hexdigest())
    lvos = [c for c in order if store.lvos[c]][:count // 2]
    rest = [c for c in order if not store.lvos[c]][:count - len(lvos)]
    picks = [(c, OFFSETS[k % len(OFFSETS)]) for k, c in enumerate(lvos + rest)]
    with ThreadPoolExecutor(6) as pool:
        items = list(pool.map(lambda job: store.item(*job), picks))
    return [i for i in items if i is not None]


@torch.no_grad()
def measure(teacher, translate, items, device, pointer=None):
    row = {}
    for scenario in ('cond', 'alone', 'all'):
        sums = dict(bad=0)
        for item in items:
            terms, info = one(teacher, translate, item, scenario, device, pointer)
            for key, value in {**terms, **info}.items():
                sums[key] = sums.get(key, 0.) + float(value)
            sums['bad'] += info['iou'] < .5
        row[scenario] = {k: round(v / len(items), 5) for k, v in sums.items()}
    return row


def fixed_translators(device, names=None):
    ridge = torch.load(ROOT / 'runs/ridge/ridge.pt', map_location=device, weights_only=True)
    w, b = ridge['spatial_w'], ridge['spatial_b']
    raw = torch.load(LEGACY, map_location='cpu', weights_only=True)
    legacy = restore_model('transformer', raw.get('transformer', raw)).to(device).float().eval()
    zeros = torch.zeros(1, 1, 1, 256, device=device)

    def gain(function, g):
        def run(x):
            a = function(x)
            c = a.mean((2, 3), keepdim=True)
            return c + g * (a - c)
        return run
    table = {'ridge': lambda x: x.movedim(1, -1).double().matmul(w).add(b).movedim(-1, 1).float(),
             'legacy': lambda x: legacy.translate_tensors(x[None, None], zeros.expand(1, 1, x.shape[0], 256))[0][0, 0]}
    for name in [n for n in names or ('mlp', 'mlp_context', 'ds3') if n not in table]:
        saved = torch.load(ROOT / 'runs/spatial' / name / 'model.pt', map_location='cpu', weights_only=False)
        model = make_spatial(saved['name'], ridge)
        model.load_state_dict(saved['state'])
        table[name] = model.to(device).float().eval()
    if names:
        return {name: table[name] for name in names}
    rows = dict(table)
    for name in ('legacy', 'mlp'):
        for g in (1.1, 1.2, 1.3, 1.5):
            rows[f'{name}*{g}'] = gain(table[name], g)
    for name in ('ridge', 'mlp_context', 'ds3'):
        rows[f'{name}*1.1'] = gain(table[name], 1.1)
    return rows


def probe(args, teacher, device):
    store = Store('dev')
    items = held_out(store, args.held_out)
    print(len(items), 'items', flush=True)
    names = args.probe_models.split(',') if args.probe_models else None
    pointer = ridge_pointer(device) if args.ridge_pointer else None
    for name, translate in fixed_translators(device, names).items():
        started = time.monotonic()
        row = measure(teacher, translate, items, device, pointer)
        text = ' | '.join(f"{s} feat {r['feature']:.5f} iou {r['iou']:.4f} bad {r['bad']:.3f} lost {r['lost']:.4f} extra {r['extra']:.4f} gap {r['gap']:+.3f}"
                          for s, r in row.items())
        print(f'{name:16s} {text} | {time.monotonic() - started:.0f}s', flush=True)


def train(args, teacher, device):
    out = ROOT / 'runs/spatial' / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        print('exists', args.tag)
        return
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    saved = torch.load(ROOT / 'runs/spatial' / args.init / 'model.pt', map_location='cpu', weights_only=False)
    ridge = torch.load(ROOT / 'runs/ridge/ridge.pt', map_location='cpu', weights_only=True)
    model = make_spatial(saved['name'], ridge)
    model.load_state_dict(saved['state'])
    model = model.to(device).float().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    start, history, best = 0, [], -math.inf
    if (out / 'last.pt').exists():
        last = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(last['state'])
        optimizer.load_state_dict(last['optimizer'])
        start, history, best = last['step'], last['history'], last['best']
        print('resumed at', start, flush=True)
    items = held_out(Store('dev'), args.held_out)
    feed = Feed(Store('fit'), args.seed + start, lvos_share=args.lvos_share)
    pointer = ridge_pointer(device) if args.ridge_pointer else None
    print(len(items), 'held-out items', sum(p.numel() for p in model.parameters()), 'parameters', flush=True)

    def validate(step):
        model.eval()
        row = dict(step=step, dev=measure(teacher, model, items, device, pointer))
        model.train()
        history.append(row)
        print(json.dumps(row), flush=True)
        return row['dev']['cond']['iou']

    if start == 0:
        first = validate(0)
        best = first
    started = time.monotonic()
    sums = {}
    for step in range(start, args.updates):
        lr = args.lr * min(1, (step + 1) / 200) * .5 * (1 + math.cos(math.pi * step / args.updates))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.batch):
            item = feed.next()
            scenario = 'alone' if rng.random() < args.alone_share else 'cond' if rng.random() < args.cond_share else 'all'
            terms, info = one(teacher, model, item, scenario, device, pointer)
            loss = (args.w_feature * terms['feature'] + args.w_mask * terms['mask'] + args.w_score * terms['score']
                    + args.w_logit * terms['logit'] + args.w_dice * terms['dice'] + args.lam * terms['state'])
            (loss / args.batch).backward()
            for key, value in {**terms, **info}.items():
                sums[key] = sums.get(key, 0.) + float(value)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        done = step + 1
        if done % args.log_every == 0:
            n = args.log_every * args.batch
            print(json.dumps(dict(step=done, lr=round(lr, 7), seconds=round(time.monotonic() - started, 1), **{k: round(v / n, 5) for k, v in sums.items()})), flush=True)
            sums = {}
        if done % args.val_every == 0 or done == args.updates:
            score = validate(done)
            if score > best:
                best = score
                torch.save(dict(name=saved['name'], state=model.state_dict()), out / 'best.pt')
            torch.save(dict(state=model.state_dict(), optimizer=optimizer.state_dict(), step=done, history=history, best=best), out / 'last.pt')
    # J&F uses the last weights: one item moves the held-out IoU by .004, so its best step is mostly noise.
    torch.save(dict(name=saved['name'], state=model.state_dict()), out / 'model.pt')
    (out / 'result.json').write_text(json.dumps(dict(args=vars(args), history=history, best=best, seconds=time.monotonic() - started), indent=1))
    print(args.tag, 'finished best', best, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--init', default='mlp')
    parser.add_argument('--updates', type=int, default=12000)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--cond-share', type=float, default=.7)
    parser.add_argument('--alone-share', type=float, default=0., help='chance of the alone scenario; the rest splits by --cond-share')
    parser.add_argument('--w-feature', type=float, default=1.)
    parser.add_argument('--w-mask', type=float, default=1.)
    parser.add_argument('--w-score', type=float, default=.1)
    parser.add_argument('--w-dice', type=float, default=0.)
    parser.add_argument('--w-logit', type=float, default=0., help='weight of a Huber loss on the object score logit')
    parser.add_argument('--lvos-share', type=float, default=0., help='chance of drawing an LVOS case; 0 draws cases uniformly')
    parser.add_argument('--lam', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--held-out', type=int, default=256)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--val-every', type=int, default=1000)
    parser.add_argument('--ridge-pointer', action='store_true', help='translated records carry ridge pointers instead of target pointers')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--probe-models', help='comma list: ridge, legacy or folders of runs/spatial')
    args = parser.parse_args()
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('E2E_GPU_FRACTION', .3)))
    teacher = Teacher(device)
    if args.probe:
        probe(args, teacher, device)
    else:
        train(args, teacher, device)


if __name__ == '__main__':
    main()
