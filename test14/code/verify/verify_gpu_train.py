#!/usr/bin/env python3
"""Run the repository trainer on CUDA, with its deterministic settings, on a small real sample.

The CPU-only guard for sample indexes is lifted in memory only, so the real device path
runs. Results stay under VERIFY_DIR and are behaviour checks, never official scores.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from vos_memory_inspector.models._spatial import SAM21_MEMORY_SPEC as SPEC
from vos_memory_inspector.state_training.data import load_index
from vos_memory_inspector.state_training.determinism import configure_determinism
from vos_memory_inspector.state_training.policy import TrainConfig
from vos_memory_inspector.state_training.preflight import inspect
from vos_memory_inspector.state_training.ridge import fit_ridge
from vos_memory_inspector.state_training.trainer import construct_model, train

WORK = Path(os.environ['VERIFY_DIR'])
OUT = WORK / os.environ.get('GPU_RUN_NAME', 'gpu-train')
SRC = Path(os.environ['CMMT_PREP_ROOT']) / 'source'
ROOT_MAP = {'/workspace/CMMT-task07-artifacts/production':
            '/workspace/j700qtcgrg/CMMT-task07-artifacts/production'}
MODELS = ('residual_mlp', 'transformer', 'mlp_context')


def raw_config(name):
    return json.loads((SRC / 'configs/state_training' / f'{name}.json').read_text())


def direct():
    """Smallest reproduction: one forward/backward per model under the trainer's settings."""
    profile = configure_determinism('cuda')
    print('profile', profile, flush=True)
    ok = {}
    for name in MODELS:
        torch.manual_seed(7)
        model = construct_model(name, SPEC, raw_config(name)['model_options']).to('cuda', dtype=torch.float32)
        x = torch.randn(4, 64, 64, 64, device='cuda')
        p = torch.randn(4, 256, device='cuda')
        try:
            s, q = model.translate_tensors(x[None, None], p[None, None])
            forward_ok = True
            (s.square().mean() + q.square().mean()).backward()
            ok[name] = dict(forward=True, backward=True, error=None)
        except RuntimeError as exc:
            ok[name] = dict(forward=locals().get('forward_ok', False), backward=False, error=str(exc)[:600])
        print(name, ok[name], flush=True)
        del model
    return ok


def trainer_runs():
    index = load_index(WORK / os.environ['SAMPLE_INDEX'], ROOT_MAP, allow_sample=True)
    training = raw_config('transformer')['training']
    config = TrainConfig(**{**training, 'workers': 2, 'pin_memory': True, 'max_epochs': 2, 'min_epochs': 1})
    index.official = True  # In-memory only: selects the CUDA path in train().
    report = inspect(index, config)
    print('ready', report['full_data_ready'], 'invalid', report['invalid'][:2], 'cases', report['checked_cases'], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ridge = OUT / 'ridge.pth'
    fit_ridge(index, config, ridge)
    results = {}
    for name in MODELS:
        options = raw_config(name)['model_options']
        affine = options.get('skip_mode', 'identity') == 'affine'
        started = time.monotonic()
        try:
            torch.cuda.reset_peak_memory_stats()
            model, history = train(index, config, name, OUT / name, options=options, ridge_path=ridge if affine else None,
                                   device='cuda', readiness=report)
            results[name] = dict(ok=True, epochs=len(history), device=str(next(model.parameters()).device),
                                 validation=[h['validation']['all']['state_loss'] for h in history],
                                 seconds=round(time.monotonic() - started, 1),
                                 peak_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1))
        except Exception as exc:
            results[name] = dict(ok=False, error=f'{type(exc).__name__}: {str(exc)[:700]}',
                                 trace=traceback.format_exc()[-900:])
        print(name, results[name], flush=True)
    return results


def ddp_runs():
    """Two-process NCCL run, as in the documented launch; each rank owns one GPU."""
    import torch.distributed as dist
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    configure_determinism('cuda')
    dist.init_process_group(backend='nccl')
    index = load_index(WORK / os.environ['SAMPLE_INDEX'], ROOT_MAP, allow_sample=True)
    training = raw_config('transformer')['training']
    config = TrainConfig(**{**training, 'workers': 2, 'pin_memory': True, 'max_epochs': 2, 'min_epochs': 1})
    index.official = True  # In-memory only: selects the CUDA path in train().
    report = inspect(index, config)
    failures = []
    for name in MODELS:
        options = raw_config(name)['model_options']
        affine = options.get('skip_mode', 'identity') == 'affine'
        try:
            model, history = train(index, config, name, OUT / 'ddp' / name, options=options,
                                   ridge_path=os.environ['RIDGE'] if affine else None,
                                   device=f'cuda:{local_rank}', readiness=report)
            values = [h['validation']['all']['state_loss'] for h in history]
            digest = float(sum(p.detach().double().abs().sum() for p in model.parameters()))
            print(f'rank={dist.get_rank()} {name} OK epochs={len(history)} validation={values} '
                  f'device={next(model.parameters()).device} weight_sum={digest:.6f}', flush=True)
        except Exception as exc:
            failures.append(name)
            print(f'rank={dist.get_rank()} {name} FAILED {type(exc).__name__}: {str(exc)[:700]}', flush=True)
            traceback.print_exc()
    dist.destroy_process_group()
    return 1 if failures else 0


def main():
    if '--ddp' in sys.argv:
        return ddp_runs()
    print(torch.__version__, torch.cuda.get_device_name(0), flush=True)
    outcome = dict(torch=torch.__version__, direct=direct())
    if '--direct-only' not in sys.argv:
        outcome['trainer'] = trainer_runs()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(outcome, ensure_ascii=False, indent=1))
    failed = [k for k, v in outcome['direct'].items() if not v['backward']] + \
             [k for k, v in outcome.get('trainer', {}).items() if not v['ok']]
    print('FAILED MODELS:' if failed else 'ALL OK', failed, flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
