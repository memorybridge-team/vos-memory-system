"""Native/replay control, per-frame outputs, retained-state and preflight audits."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import random
import sys
import os
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runtime'
SAM = Path(os.environ.get('SAM2_ROOT', HERE.parent.parent / 'sam2')).expanduser().resolve()
sys.path[:0] = [str(SAM), str(ROOT)]
import numpy as np
import torch
from PIL import Image
import settings
from baseline.no_handoff import full_replay
from evaluation.scoring.jf import prepare_ground_truth, score_prepared, j_score
from evaluation.switches import switch_points
from model.sam2_runner import SAM2Runner, MEMORY_FIELDS


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):
    h = hashlib.sha256()
    h.update(str((a.shape, str(a.dtype))).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def flatten(value, prefix='', arrays=None, metadata=None):
    arrays = {} if arrays is None else arrays
    metadata = {} if metadata is None else metadata
    if isinstance(value, torch.Tensor):
        metadata[prefix+'/@dtype'] = str(value.dtype)
        metadata[prefix+'/@shape'] = list(value.shape)
        arrays[prefix] = value.detach().float().cpu().numpy().copy()
    elif isinstance(value, np.ndarray):
        arrays[prefix] = value.copy()
        metadata[prefix+'/@dtype'] = str(value.dtype)
        metadata[prefix+'/@shape'] = list(value.shape)
    elif isinstance(value, dict):
        metadata[prefix+'/@keys'] = sorted(str(k) for k in value)
        for key in sorted(value, key=str):
            flatten(value[key], prefix+'/'+str(key), arrays, metadata)
    elif isinstance(value, (list, tuple)):
        metadata[prefix+'/@length'] = len(value)
        for i, item in enumerate(value):
            flatten(item, prefix+'/'+str(i), arrays, metadata)
    elif value is None or isinstance(value, (str, bool, int, float)):
        metadata[prefix] = value
    elif isinstance(value, torch.device):
        metadata[prefix] = str(value)
    else:
        raise TypeError(f'Unexamined state value: {prefix}: {type(value)}')
    return {'arrays': arrays, 'metadata': metadata}


def state_snapshot(session):
    # LazyFrames is an input reader, not a prediction-state tensor. RGB hashes are
    # audited separately in the manifest. Everything else in inference_state is included.
    return flatten({k: v for k, v in session.state.items() if k != 'images'})


def output_snapshot(session, out, full_logits):
    return flatten({'binary_mask': out.mask, 'full_resolution_logits': full_logits,
                    **{k: session.memory_of(out.frame)[k] for k in MEMORY_FIELDS}})


def hashes(snapshot):
    return {'metadata': snapshot['metadata'],
            'arrays': {k: array_sha(v) for k, v in snapshot['arrays'].items()}}


def comparison(a, b):
    metadata_equal = a['metadata'] == b['metadata']
    keys_equal = a['arrays'].keys() == b['arrays'].keys()
    fields = {}
    for key in a['arrays'].keys() & b['arrays'].keys():
        x, y = a['arrays'][key], b['arrays'][key]
        if x.shape != y.shape or x.dtype != y.dtype:
            fields[key] = {'exact': False, 'shape_dtype_equal': False}
            continue
        finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
        exact = bool(np.array_equal(x, y)) and finite
        delta = None if exact else x.astype(np.float64)-y.astype(np.float64)
        fields[key] = {'exact': exact, 'finite': finite,
                       'different_elements': 0 if exact else int(np.count_nonzero(x != y)),
                       'max_abs_error': 0.0 if exact else float(np.abs(delta).max()),
                       'mean_abs_error': 0.0 if exact else float(np.abs(delta).mean()),
                       'rmse': 0.0 if exact else float(np.sqrt(np.square(delta).mean())),
                       'allclose_1e5_1e6': bool(finite and (exact or np.allclose(x, y, rtol=1e-5, atol=1e-6)))}
    return {'exact': metadata_equal and keys_equal and all(f['exact'] for f in fields.values()),
            'metadata_equal': metadata_equal, 'array_keys_equal': keys_equal,
            'fields': dict(sorted(fields.items()))}


def seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def parameter_sha(predictor):
    h = hashlib.sha256()
    for key, value in predictor.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'mps'], required=True)
    parser.add_argument('--frames', type=int, default=33)
    parser.add_argument('--videos', nargs='+', default=['bmx-bumps', 'camel', 'breakdance'])
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--output-name')
    args = parser.parse_args()
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(args.deterministic)
    settings.DEVICE = args.device
    settings.USE_BF16 = False
    settings.PREFETCH_FRAMES = False
    settings.SAM2_CHECKPOINT_DIR = str(SAM/'checkpoints')
    import sam2.sam2_video_predictor as svp
    svp.tqdm = lambda it, **kwargs: it
    dest = HERE/'extended'/(args.output_name or ('rerun-'+args.device))
    if dest.resolve() in {(HERE/'extended/cpu').resolve(), (HERE/'extended/mps').resolve()}:
        parser.error('Recorded evidence is protected; choose a different --output-name')
    dest.mkdir(parents=True, exist_ok=True)
    seed()
    runner = SAM2Runner('base_plus')
    initial_parameter_sha = parameter_sha(runner.predictor)
    environment = {'device': args.device, 'torch': torch.__version__,
                   'platform': platform.platform(), 'frames': args.frames,
                   'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                   'checkpoint_sha256': file_sha(SAM/'checkpoints/sam2.1_hiera_base_plus.pt'),
                   'parameter_sha256': initial_parameter_sha,
                   'predictor_sha256': file_sha(SAM/'sam2/sam2_video_predictor.py'),
                   'test_script_sha256': file_sha(__file__),
                   'session_sha256': file_sha(ROOT/'model/sam2_runner.py'),
                   'full_replay_sha256': file_sha(ROOT/'baseline/no_handoff.py'),
                   'seed': 0, 'inference_precision': 'FP32; SAM2 stored memory BF16',
                   'images_excluded_from_state_snapshot': 'LazyFrames reader; exact RGB SHA256 recorded separately',
                   'tolerances': {'rtol': 1e-5, 'atol': 1e-6},
                   'postprocessing': 'CUDA _C extension absent; identical skipped hole filling'}
    (dest/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    source = SAM/'data/DAVIS'
    for name in args.videos:
        paths = sorted((source/'JPEGImages/480p'/name).glob('*.jpg'))[:args.frames]
        if len(paths) != args.frames:
            raise ValueError(f'Insufficient RGB frames: {name}')
        gt_paths = [source/'Annotations/480p'/name/(p.stem+'.png') for p in paths]
        labels = [np.array(Image.open(p)) for p in gt_paths]
        obj = int(min(set(np.unique(labels[0]))-{0, 255}))
        prompt = labels[0] == obj
        if not prompt.any():
            raise ValueError('Empty initial prompt')
        h, w = prompt.shape
        with Image.open(paths[0]) as im:
            if im.size != (w, h):
                raise ValueError('GT/RGB sizes differ')
        gts = [prepare_ground_truth(a == obj, ignore=a == 255) for a in labels]
        video = SimpleNamespace(frame_paths=paths, size=(h, w))
        end = len(paths)-1
        switches = switch_points(0, end)
        if len(switches) != 3:
            raise ValueError('Need 25/50/75 switch points with at least 8 prefix frames')
        report = {'video': name, 'original_object_id': obj, 'session_object_id': 1,
                  'frame_count': len(paths), 'resolution': [h, w], 'switches': switches,
                  'input_rgb_sha256': {p.name: file_sha(p) for p in paths},
                  'input_gt_sha256': {p.name: file_sha(p) for p in gt_paths},
                  'prompt_sha256': array_sha(prompt), 'runs': []}
        references = {}
        reference_states = {}
        reference_scores = {}
        native_control = {}
        control_states = {}
        first_ids = []
        variants = [('native_0', None), ('native_1', None), ('full_replay_continuous', None)]
        variants += [('full_replay_'+sw['name'], sw['frame']) for sw in switches]
        for variant, switch in variants:
            seed()
            session = runner.start(video)
            # State identities are retained until video completion to test non-reuse.
            first_ids.append(session.state)
            if any(session.state is old for old in first_ids[:-1]):
                raise AssertionError('Inference state reused')
            session.configure_benchmark()
            run = {'variant': variant, 'switch_frame': switch, 'frames': [],
                   'switch_states': {}, 'preflight_restart_audit': None,
                   'single_frame_inference_calls': [], 'memory_encoder_calls': []}
            predictor = runner.predictor
            originals = {k: getattr(predictor, k) for k in
                         ['propagate_in_video', '_run_single_frame_inference', '_run_memory_encoder']}
            latest = {}

            def observed_propagate(*a, **kw):
                for frame, obj_ids, logits in originals['propagate_in_video'](*a, **kw):
                    latest['full_logits'] = logits.detach().float().cpu().numpy().copy()
                    if obj_ids != [1]:
                        raise AssertionError('Object mapping changed')
                    yield frame, obj_ids, logits

            def observed_inference(*a, **kw):
                run['single_frame_inference_calls'].append(
                    {k: kw.get(k) for k in ['frame_idx', 'reverse', 'run_mem_encoder', 'is_init_cond_frame']})
                return originals['_run_single_frame_inference'](*a, **kw)

            def observed_memory(*a, **kw):
                run['memory_encoder_calls'].append(kw['frame_idx'])
                return originals['_run_memory_encoder'](*a, **kw)

            predictor.propagate_in_video = observed_propagate
            predictor._run_single_frame_inference = observed_inference
            predictor._run_memory_encoder = observed_memory
            t0 = time.perf_counter()
            try:
                if variant.startswith('native'):
                    session.add_prompt(0, prompt.copy())
                    first = 0
                else:
                    first = full_replay(session, 0, prompt.copy())
                session.encode_prompts()
                segments = [(first, end)] if switch is None else [(first, switch), (switch+1, end)]
                for segment_index, (lo, hi) in enumerate(segments):
                    if segment_index:
                        before = state_snapshot(session)
                        session.encode_prompts()
                        after = state_snapshot(session)
                        run['preflight_restart_audit'] = comparison(before, after)
                        del before, after
                    for out in session.track(lo, hi):
                        f = out.frame
                        current = output_snapshot(session, out, latest['full_logits'])
                        score = score_prepared(out.mask, gts[f])
                        row = {'frame': f, 'gt_j': score.j, 'gt_f': score.f, 'gt_jf': score.jf,
                               'output_hashes': hashes(current)}
                        if variant == 'native_0':
                            references[f] = current
                            reference_scores[f] = score
                        else:
                            row['vs_native_0'] = comparison(references[f], current)
                            row['mask_iou_vs_native_0'] = j_score(current['arrays']['/binary_mask'],
                                                                references[f]['arrays']['/binary_mask'])
                            row['gt_jf_difference_vs_native_0'] = score.jf-reference_scores[f].jf
                            if variant != 'native_1':
                                row['vs_native_1'] = comparison(native_control[f], current)
                                row['mask_iou_vs_native_1'] = j_score(current['arrays']['/binary_mask'],
                                                                    native_control[f]['arrays']['/binary_mask'])
                        if variant == 'native_1':
                            native_control[f] = current
                        if f in {s['frame'] for s in switches}:
                            entire = state_snapshot(session)
                            run['switch_states'][str(f)] = {'hashes': hashes(entire)}
                            if variant == 'native_0':
                                reference_states[f] = entire
                            else:
                                run['switch_states'][str(f)]['vs_native_0'] = comparison(reference_states[f], entire)
                                if variant != 'native_1':
                                    run['switch_states'][str(f)]['vs_native_1'] = comparison(control_states[f], entire)
                            if variant == 'native_1':
                                control_states[f] = entire
                        run['frames'].append(row)
                        print(f'{args.device} {name} {variant} frame={f}', flush=True)
                if [r['frame'] for r in run['frames']] != list(range(end+1)):
                    raise AssertionError('Frame coverage incomplete')
            finally:
                for key, original in originals.items():
                    setattr(predictor, key, original)
                session.close()
                # Prove distinct identity without retaining large GPU state tensors.
                first_ids[-1].clear()
            run['seconds_with_audit'] = time.perf_counter()-t0
            if report['runs']:
                run['inference_schedule_matches_native_0'] = (
                    run['single_frame_inference_calls'] == report['runs'][0]['single_frame_inference_calls'] and
                    run['memory_encoder_calls'] == report['runs'][0]['memory_encoder_calls'])
            report['runs'].append(run)
            (dest/(name+'.json')).write_text(json.dumps(report, indent=2)+'\n')
            print(f'COMPLETE {args.device} {name} {variant}', flush=True)
        report['all_runs_have_independent_states'] = True
        report['complete'] = len(report['runs']) == 6
        (dest/(name+'.json')).write_text(json.dumps(report, indent=2)+'\n')
        del references, reference_states, native_control, control_states, first_ids
    final_sha = parameter_sha(runner.predictor)
    environment['parameter_sha256_after'] = final_sha
    environment['model_weights_unchanged'] = final_sha == initial_parameter_sha
    environment['complete'] = True
    (dest/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    if final_sha != initial_parameter_sha:
        raise AssertionError('Model weights changed')


if __name__ == '__main__':
    main()
