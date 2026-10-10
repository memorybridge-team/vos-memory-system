"""Independent checkpoint-backed Native vs segmented Full Replay comparison."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SAM = ROOT.parent / 'sam2'
sys.path[:0] = [str(SAM), str(ROOT)]
import numpy as np
import torch
from PIL import Image
import settings
from model.sam2_runner import SAM2Runner, MEMORY_FIELDS
from baseline.no_handoff import full_replay
from evaluation.switches import switch_points


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(session, out):
    data = {'binary_mask': out.mask.copy()}
    for key in MEMORY_FIELDS:
        value = session.memory_of(out.frame)[key]
        values = value if isinstance(value, (tuple, list)) else [value]
        for i, item in enumerate(values):
            # NumPy has no native bfloat16. FP32 conversion is exact for BF16 values.
            data[f'{key}/{i}'] = None if item is None else item.detach().float().cpu().numpy().copy()
    return data


def compare(a, b):
    if set(a) != set(b):
        raise ValueError('Output fields differ')
    result = {}
    for key in a:
        x, y = a[key], b[key]
        if x is None or y is None:
            result[key] = {'equal': x is None and y is None, 'max_abs_error': None}
        else:
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError(f'Shape/dtype differs: {key}')
            result[key] = {'equal': bool(np.array_equal(x, y)),
                           'max_abs_error': float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))),
                           'different_elements': int(np.count_nonzero(x != y))}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='mps', choices=['cpu', 'mps'])
    parser.add_argument('--frames', type=int, default=33)
    parser.add_argument('--min-pre-frames', type=int, default=8)
    parser.add_argument('--control-only', action='store_true',
                        help='Compare two uninterrupted Native runs instead of segmented replay')
    args = parser.parse_args()
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    settings.DEVICE = args.device
    settings.USE_BF16 = False
    settings.SAM2_CHECKPOINT_DIR = str(SAM / 'checkpoints')
    settings.PREFETCH_FRAMES = False
    torch.set_num_threads(4)
    torch.manual_seed(0)
    paths = sorted((SAM / 'notebooks/videos/bedroom').glob('*.jpg'))[:args.frames]
    if len(paths) != args.frames:
        raise ValueError('Insufficient frames')
    with Image.open(paths[0]) as im:
        w, h = im.size
    video = SimpleNamespace(frame_paths=paths, size=(h, w))
    fixture = ROOT / 'test11/fixture_cpu.npz'
    with np.load(fixture) as data:
        prompt = data['initial'].copy()
    if prompt.shape != (h, w):
        raise ValueError('Prompt shape differs')
    end = len(paths)-1
    switches = ([{'name': '100', 'frame': end}] if args.control_only else
                switch_points(0, end, min_pre_frames=args.min_pre_frames))
    runner = SAM2Runner('base_plus')
    report = {'device': args.device, 'inference_dtype': 'float32',
              'memory_storage_dtype': 'SAM2 internally casts maskmem_features to bfloat16',
              'torch': torch.__version__,
              'platform': platform.platform(), 'video': str(paths[0].parent),
              'frame_count': len(paths), 'object_count': 1,
              'min_pre_frames': args.min_pre_frames,
              'checkpoint_sha256': sha(SAM/'checkpoints/sam2.1_hiera_base_plus.pt'),
              'predictor_sha256': sha(SAM/'sam2/sam2_video_predictor.py'),
              'prompt_fixture_sha256': sha(fixture),
              'frame_sha256': {p.name: sha(p) for p in paths},
              'prompt_source': 'test11 Small prediction; no ground-truth accuracy evaluation',
              'control_only': args.control_only, 'comparisons': []}
    native = {}
    session = runner.start(video)
    started = time.perf_counter()
    try:
        session.configure_benchmark()
        session.add_prompt(0, prompt)
        for out in session.track(0, end):
            native[out.frame] = snapshot(session, out)
            print(f'native frame={out.frame}', flush=True)
    finally:
        session.close()
    if set(native) != set(range(end+1)):
        raise ValueError('Native frame coverage incomplete')
    report['native_seconds'] = time.perf_counter()-started
    for sw in switches:
        torch.manual_seed(0)
        session = runner.start(video)
        started = time.perf_counter()
        rows = []
        try:
            session.configure_benchmark()
            first = full_replay(session, 0, prompt)
            # Exhaust the prefix iterator, then start a new continuation iterator.
            for lo, hi in [(first, sw['frame']), (sw['frame']+1, end)]:
                if lo > hi:
                    continue
                for out in session.track(lo, hi):
                    rows.append({'frame': out.frame,
                                 'phase': 'prefix' if out.frame <= sw['frame'] else 'future',
                                 'fields': compare(native[out.frame], snapshot(session, out))})
                    print(f'replay switch={sw["name"]} frame={out.frame}', flush=True)
        finally:
            session.close()
        if [r['frame'] for r in rows] != list(range(end+1)):
            raise ValueError('Replay frame coverage incomplete')
        report['comparisons'].append({'switch_percent': int(sw['name']),
                                     'switch_frame': sw['frame'],
                                     'future_frames': end-sw['frame'],
                                     'seconds': time.perf_counter()-started, 'frames': rows})
        report['all_equal'] = all(f['equal'] for c in report['comparisons'] for r in c['frames'] for f in r['fields'].values())
        filename = f'control_{args.device}.json' if args.control_only else f'results_{args.device}.json'
        (HERE/filename).write_text(json.dumps(report, indent=2)+'\n')
        print(f'completed switch={sw["name"]} all_equal={report["all_equal"]}', flush=True)
    if len(switches) != (1 if args.control_only else 3):
        raise ValueError('Need all three eligible switch points')
    if not report['all_equal']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
