#!/usr/bin/env python3
"""Evaluate frozen Affine on a deterministic fraction of a prepared state bank.

No private test9 runtime modules are required. Target-state continuation is a
reference from the saved Base+ state, not a claimed full-prefix native replay.
Missing GT remains missing. Memory scores cover valid whole records only.
"""
import argparse
from collections import OrderedDict, defaultdict
import dataclasses
import json
import math
from pathlib import Path
import sys
import time

DEADLINE = float('inf')


def check_deadline():
    if time.time() >= DEADLINE:
        raise TimeoutError('evaluation wall-time budget exhausted')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import read, write, sha, rank
from manifest import library_imports
library_imports()
import numpy as np
import torch
from PIL import Image
from state_pairs import load_pair, active_state
from training import load_models


def select(args, training):
    roles = {(r['dataset'], r['video_id']): r['split'] for r in training['collection']['pairs']}
    selected, counts = [], {}
    for dataset, folder in [('MOSEv2', 'mose'), ('LVOSv2', 'lvos')]:
        paths = sorted((args.bank / folder / 'cache').glob('*/*.pt'))
        n = math.ceil(len(paths) * args.fraction)
        chosen = sorted(paths, key=lambda p: rank(args.seed, dataset, p.name))[:n]
        counts[dataset] = dict(available=len(paths), selected=n)
        for path in chosen:
            meta = read(path.with_suffix('.prepare.json'))
            membership = roles.get((dataset, meta['video_id']))
            selected.append(dict(path=str(path), dataset=dataset, video_id=meta['video_id'],
                object_id=int(meta['object_id']), switch=int(meta['switch_frame']),
                num_frames=int(meta['num_frames']), sha256=meta['cache']['sha256'],
                role={'train':'training_video', 'validation':'checkpoint_selection_video'}.get(
                    membership, 'excluded_from_training_and_selection')))
    groups=[[r for r in selected if r['dataset']==d] for d in ['MOSEv2','LVOSv2']]
    interleaved=[g[i] for i in range(max(map(len,groups))) for g in groups if i<len(g)]
    return dict(schema='test10.affine_storage_selection.v1', seed=args.seed,
                fraction=args.fraction, counts=counts, pairs=interleaved)


def memory(source, target, model, scales):
    valid = source.validity
    x, p, y, q = [t[valid].to('cuda', dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                  for t in (source.spatial_memory, source.object_pointer,
                            target.spatial_memory, target.object_pointer)]
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        a, b = model.translate_tensors(x, p)
        result = {}
        for name, u, v in [('direct', x, p), ('affine_fp32', a, b),
                           ('affine_handoff', a.bfloat16().float(), b)]:
            spatial = float((u-y).square().mean())
            pointer = float((v-q).square().mean())
            result[name] = dict(spatial_mse=spatial, pointer_mse=pointer,
                normalized_loss=spatial/scales['spatial']**2+pointer/scales['pointer']**2)
    return dict(records=int(valid.sum()), methods=result)


class LazyFrames:
    """Same pinned SAM2 JPEG resize/normalization, without whole-video RAM use."""
    def __init__(self, paths, size):
        self.paths, self.size = paths, size
        with Image.open(paths[0]) as image:
            self.width, self.height = image.size
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from sam2.utils.misc import _load_img_as_tensor
        if index not in self.cache:
            image, h, w = _load_img_as_tensor(str(self.paths[index]), self.size)
            if (h, w) != (self.height, self.width):
                raise ValueError('variable video dimensions')
            image = image.float()
            image.sub_(torch.tensor([.485,.456,.406])[:,None,None])
            image.div_(torch.tensor([.229,.224,.225])[:,None,None])
            self.cache[index] = image
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        return self.cache[index]


def fresh(predictor, frames):
    return dict(images=frames, num_frames=len(frames), offload_video_to_cpu=True,
        offload_state_to_cpu=True, video_height=frames.height, video_width=frames.width,
        device=predictor.device, storage_device=torch.device('cpu'),
        point_inputs_per_obj={}, mask_inputs_per_obj={}, cached_features={}, constants={},
        obj_id_to_idx=OrderedDict(), obj_idx_to_id=OrderedDict(), obj_ids=[],
        output_dict_per_obj={}, temp_output_dict_per_obj={}, frames_tracked_per_obj={})


def video(args, row):
    name = 'MOSEv2' if row['dataset']=='MOSEv2' else 'LVOSv2'
    matches = [root for root in (args.data / name / 'extracted').iterdir()
               if (root/'JPEGImages'/row['video_id']).is_dir()
               and (root/'Annotations'/row['video_id']).is_dir()]
    if len(matches)!=1:
        raise ValueError(f'ambiguous/missing video {row["video_id"]}')
    rgb, gt = [matches[0]/folder/row['video_id'] for folder in ['JPEGImages','Annotations']]
    paths = sorted([p for p in rgb.iterdir() if p.suffix.lower() in ('.jpg','.jpeg')],
                   key=lambda p: int(p.stem))
    if len(paths)!=row['num_frames']:
        raise ValueError('pair/video length mismatch')
    stride = 5 if row['dataset']=='LVOSv2' else 1
    if any(int(b.stem)-int(a.stem)!=stride for a,b in zip(paths,paths[1:])):
        raise ValueError('frame stride mismatch')
    return paths, gt


def continuation(predictor, frames, canonical, row, end, gt, paths, metrics):
    from vos_memory_inspector.sam2_state import inject_sam2_canonical_state
    from vos_memory_inspector.vos_metrics import read_indexed_png
    state = fresh(predictor, frames)
    calls, current = [], [None]
    original_feature, original_forward = predictor._get_image_feature, predictor.forward_image
    def feature(s, frame_idx, batch_size):
        current[0] = int(frame_idx)
        if frame_idx <= row['switch']:
            raise RuntimeError('past RGB accessed during handoff continuation')
        return original_feature(s, frame_idx, batch_size)
    def forward(image):
        calls.append(current[0])
        return original_forward(image)
    predictor._get_image_feature, predictor.forward_image = feature, forward
    scores = []
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            inject_sam2_canonical_state(canonical, predictor=predictor, inference_state=state)
            if calls:
                raise RuntimeError('injection ran the backbone')
            for position, ids, logits in predictor.propagate_in_video(state,
                    start_frame_idx=row['switch']+1, max_frame_num_to_track=end-row['switch']-1):
                check_deadline()
                if position>end:
                    raise RuntimeError('unexpected prediction position')
                if len(ids)!=1 or str(ids[0])!=str(row['object_id']):
                    raise RuntimeError('object identity mismatch')
                mask = (logits[0,0]>0).cpu().numpy()
                annotation = gt / (paths[position].stem+'.png')
                item = dict(position=position, offset=position-row['switch'], J=None,F=None,J_and_F=None,
                            gt_present=None, status='missing_annotation')
                if annotation.is_file():
                    label = read_indexed_png(annotation)
                    target, void = label==row['object_id'], label==255
                    j = float(metrics.db_eval_iou(target, mask, void))
                    f = float(metrics.db_eval_boundary(target, mask, void))
                    item.update(J=j,F=f,J_and_F=(j+f)/2,gt_present=bool(target.any()),status='ok')
                scores.append(item)
    finally:
        predictor._get_image_feature, predictor.forward_image = original_feature, original_forward
    if [r['position'] for r in scores]!=list(range(row['switch']+1,end+1)):
        raise RuntimeError('incomplete continuation')
    return dict(frames=scores, past_backbone_calls=0, future_backbone_calls=len(calls))


def averages(rows):
    valid = [r for r in rows if r['status']=='ok']
    visible=[r for r in valid if r['gt_present']]
    return dict(predicted_frames=len(rows), annotated_frames=len(valid),
        visible_frames=len(visible), visible_J_and_F=sum(r['J_and_F'] for r in visible)/len(visible) if visible else None,
        **{k: sum(r[k] for r in valid)/len(valid) if valid else None for k in ['J','F','J_and_F']})


def native_gate(predictor, frames, source, row, paths, gt):
    """Check actual native Base+ vs exported/reinjected future full logits."""
    from vos_memory_inspector.sam2_state import canonicalize_sam2_inference_state, inject_sam2_canonical_state
    from vos_memory_inspector.vos_metrics import read_indexed_png
    conditioning=source.frame_indices[source.validity & source.is_conditioning].tolist()
    if len(conditioning)!=1: raise ValueError('native gate requires one conditioning prompt')
    first=int(conditioning[0])
    prompt=read_indexed_png(gt/(paths[first].stem+'.png'))==row['object_id']
    state=fresh(predictor,frames)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        predictor.add_new_mask(state,frame_idx=first,obj_id=row['object_id'],mask=prompt)
        for _ in predictor.propagate_in_video(state,start_frame_idx=first,
                max_frame_num_to_track=row['switch']-first): check_deadline()
        canonical=active_state(canonicalize_sam2_inference_state(state,switch_frame=row['switch'],strict=True),
            num_maskmem=predictor.num_maskmem,max_obj_ptrs=predictor.max_obj_ptrs_in_encoder)
        native={i:logits.float().cpu() for i,_,logits in predictor.propagate_in_video(state,
            start_frame_idx=row['switch']+1,max_frame_num_to_track=9)}
        injected=fresh(predictor,frames)
        inject_sam2_canonical_state(canonical,predictor=predictor,inference_state=injected)
        again={i:logits.float().cpu() for i,_,logits in predictor.propagate_in_video(injected,
            start_frame_idx=row['switch']+1,max_frame_num_to_track=9)}
    equal=list(native)==list(again) and all(torch.equal(native[i],again[i]) for i in native)
    if not equal: raise RuntimeError('native full-logit self-injection gate failed')
    return dict(passed=True,frames=len(native),full_logits_bitwise_equal=True,
                reference='locally regenerated Base+ prefix; separate from saved target-state baseline')


def summarize(directory):
    rows = [read(p) for p in (directory/'cases').glob('*.json')]
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset'],row['role']].append(row)
    output = {}
    for (dataset,role), cases in groups.items():
        data = dict(cases=len(cases), videos=len({c['video_id'] for c in cases}), methods={})
        for method in cases[0]['memory']['methods']:
            n = sum(c['memory']['records'] for c in cases)
            data['methods'][method] = {k:sum(c['memory']['methods'][method][k]*c['memory']['records']
                for c in cases)/n for k in ['spatial_mse','pointer_mse','normalized_loss']}
        jf = [c for c in cases if c.get('jf')]
        data['jf_complete_cases'] = len(jf)
        data['jf'] = {}
        for method in (jf[0]['jf'] if jf else {}):
            data['jf'][method] = {}
            for window in ['first10','suffix']:
                # Equal videos; equal cases within each video, excluding missing GT.
                videos = defaultdict(list)
                for case in jf:
                    value = case['jf'][method][window]['J_and_F']
                    if value is not None:
                        videos[case['video_id']].append(value)
                data['jf'][method][window] = dict(videos=len(videos),
                    J_and_F=sum(sum(v)/len(v) for v in videos.values())/len(videos) if videos else None)
        output[dataset+':'+role] = data
    write(directory/'summary.json', dict(completed_memory_cases=len(rows), groups=output))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--training-dir',type=Path,required=True)
    p.add_argument('--sam2-repo',type=Path,required=True)
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--fraction',type=float,default=.25)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--stage',choices=['memory','jf','both'],default='both')
    p.add_argument('--max-cases',type=int)
    p.add_argument('--horizon',choices=['first10','suffix'],default='suffix')
    p.add_argument('--jf-mose-cases',type=int)
    p.add_argument('--jf-lvos-cases',type=int)
    p.add_argument('--deadline-unix',type=float,help='shared absolute wall-time deadline for memory and J&F')
    args=p.parse_args()
    global DEADLINE
    DEADLINE=args.deadline_unix if args.deadline_unix else float('inf')
    if not 0<args.fraction<=1: p.error('fraction must be in (0,1]')
    torch.manual_seed(args.seed)
    model=load_models(args.training_dir,'cuda',['affine'])['affine']
    training=read(args.training_dir/'train_report.json')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    selection=select(args,training)
    selection['checkpoint_sha256']=sha(args.training_dir/'affine.pt')
    selection['evaluator_sha256']=sha(__file__)
    old=args.output_dir/'selection.json'
    if old.exists() and read(old)!=selection: raise ValueError('selection changed')
    write(old,selection)
    scales=training['normalization']['scales']
    predictor=metrics=None
    if args.stage in ('jf','both'):
        sys.path.insert(0,str(args.sam2_repo))
        from vos_memory_inspector.upstream import verify_sam2_checkout
        from vos_memory_inspector.vos_metrics import load_official_metrics
        verify_sam2_checkout(args.sam2_repo)
        metrics,reason=load_official_metrics()
        if metrics is None: raise RuntimeError(reason)
        from sam2.build_sam import build_sam2_video_predictor
        predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_b+.yaml',
                str(args.base_checkpoint),device='cuda')
        write(args.output_dir/'metric_provenance.json',metrics.to_dict())
    started=time.monotonic()
    pairs=selection['pairs'][:args.max_cases] if args.max_cases else selection['pairs']
    if args.stage=='jf' and (args.jf_mose_cases is not None or args.jf_lvos_cases is not None):
        cohorts=[]
        for dataset,limit in [('MOSEv2',args.jf_mose_cases),('LVOSv2',args.jf_lvos_cases)]:
            candidates=[r for r in selection['pairs'] if r['dataset']==dataset]
            candidates.sort(key=lambda r:rank(args.seed,'jf',dataset,r['video_id'],r['path']))
            videos=set(); chosen=[]
            for row in candidates:
                if row['video_id'] in videos: continue
                videos.add(row['video_id']);chosen.append(row)
                if limit is not None and len(chosen)>=limit: break
            cohorts.append(chosen)
        pairs=[g[i] for i in range(max(map(len,cohorts))) for g in cohorts if i<len(g)]
        write(args.output_dir/'jf_selection.json',dict(seed=args.seed,policy='one pair per video; hash order; no score selection',pairs=pairs))
    for index,row in enumerate(pairs,1):
        write(args.output_dir/'active_case.json',dict(index=index,requested=len(pairs),case=row,
            deadline_unix=args.deadline_unix))
        check_deadline()
        output=args.output_dir/'cases'/(row['dataset']+'_'+Path(row['path']).stem+'.json')
        result=read(output) if output.exists() else dict(row)
        need_memory='memory' not in result
        need_jf=args.stage in ('jf','both') and (not result.get('jf') or result.get('horizon')!=args.horizon)
        if need_memory or need_jf:
            source,target,meta=load_pair(row['path'],expected_sha=row['sha256'])
            if need_memory: result['memory']=memory(source,target,model,scales)
            if need_jf:
                paths,gt=video(args,row)
                frames=LazyFrames(paths,predictor.image_size)
                gate=args.output_dir/('native_gate_'+row['dataset']+'.json')
                if not gate.exists():
                    write(gate,native_gate(predictor,frames,source,row,paths,gt))
                end=len(paths)-1 if args.horizon=='suffix' else min(row['switch']+10,len(paths)-1)
                if end-row['switch']<10: raise ValueError('fewer than 10 future frames')
                with torch.inference_mode(),torch.autocast('cuda',enabled=False):
                    gpu=dataclasses.replace(source,spatial_memory=source.spatial_memory.to('cuda'),
                        object_pointer=source.object_pointer.to('cuda'),presence_logits=source.presence_logits.to('cuda'))
                    translated=model.translate(gpu)
                result['jf']={}
                for method,canonical in [('direct',source),('affine',translated),('target_state_reference',target)]:
                    scores=continuation(predictor,frames,canonical,row,end,gt,paths,metrics)
                    scores['first10']=averages(scores['frames'][:10])
                    scores['suffix']=averages(scores['frames']) if args.horizon=='suffix' else dict(J_and_F=None,status='not_run')
                    result['jf'][method]=scores
                result['horizon']=args.horizon
            write(output,result)
        interval=100 if args.stage=='memory' else 10
        if index%interval==0 or index==1 or index==len(pairs):
            summarize(args.output_dir)
            write(args.output_dir/'progress.json',dict(stage=args.stage,horizon=args.horizon,
                processed=index,requested=len(pairs),seconds=time.monotonic()-started,status='running'))
            print(f'{args.stage} {index}/{len(pairs)} {time.monotonic()-started:.1f}s',flush=True)
    summarize(args.output_dir)
    write(args.output_dir/'progress.json',dict(status='complete',stage=args.stage,
        horizon=args.horizon,processed=len(pairs),requested=len(pairs),seconds=time.monotonic()-started))


if __name__=='__main__': main()
