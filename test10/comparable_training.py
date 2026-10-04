#!/usr/bin/env python3
"""Fixed LVOS record-level recipe for Affine, matching the reference DDP Transformer trainer.

Reference: vos-memory-translator-nonlinear lvos_ddp.py at the recipe's reference_revision.
tools/verify_reference_recipe.py runs that trainer on Affine and compares trajectories.
"""
import argparse
from collections import OrderedDict
import copy
import io
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

from core import ROOT, REPO, read, write, sha, digest, snapshot, verify_snapshot
from training import (SCHEMA, REPORT_SCHEMA, CHECKPOINT_SCHEMA, PRESETS, fresh_model,
                      signature, training_report, torch, SAM21_MEMORY_SPEC)
from state_pairs import validate_pair
from vos_memory_inspector.state_schema import CanonicalState, StateSpec
from vos_memory_inspector.transformer_translator import TransformerStateTranslator

RECIPE_PATH = ROOT / 'configs/affine_comparable_v1.json'
INDEX_SCHEMA = 'cmmt.lvos_operational_cache_index.v1'
AFFINE_PARAMETERS = [('feature.weight', (64, 64)), ('feature.bias', (64,)),
                     ('pointer.weight', (256, 256)), ('pointer.bias', (256,))]


def recipe():
    return read(RECIPE_PATH)


def data_fingerprint(rows):
    ledger = sorted([dict(dataset=r['dataset'], video_id=r['video_id'], split=r['split'],
                          sha256=r['sha256'], valid_indices=r['valid_indices']) for r in rows],
                    key=lambda r: (r['split'], r['video_id'], r['sha256']))
    return digest(ledger)


def index_rows(document):
    return [dict(dataset='LVOSv2', video_id=e['case']['video_id'],
                 split={'fit': 'train', 'development': 'validation'}[e['case']['paired_split']],
                 path=e['path'], sha256=e['sha256'], valid_indices=e['valid_indices'],
                 valid_records=e['records'], case_id=e['case']['case_id'],
                 object_id=e['case'].get('object_id'),
                 runtime_switch_frame=e.get('runtime_switch_frame')) for e in document['entries']]


class PairData:
    """Content-verified original pairs; raw indexes need no invented prepare metadata."""
    def __init__(self, path, pair_root=None, cache_gib=2., deadline=None):
        self.path = Path(path).resolve()
        self.deadline = deadline
        input_sha = sha(self.path)
        document = read(self.path)
        self.index = document if document.get('schema_version') == INDEX_SCHEMA else None
        self.rows, self.cache, self.cache_bytes = [], OrderedDict(), 0
        if not math.isfinite(cache_gib) or cache_gib < 0:
            raise ValueError('cache-gib must be finite and nonnegative')
        self.cache_limit = int(cache_gib * 1024**3)
        if self.index is not None:
            if document.get('state') != 'tensor_validated':
                raise ValueError('tensor-validated raw index required')
            scope = document['identity']['scope']
            if scope not in ('full_frozen_fit_development', 'full_frozen_fit_development_with_exclusions'):
                raise ValueError('use the full frozen fit/development index, not a diagnostic subset')
            excluded = document['identity'].get('exclusion_plan')
            if excluded and not (excluded.get('approved_by') and excluded.get('recorded_at')):
                raise ValueError('raw index contains an unapproved exclusion plan')
            items = index_rows(document)
        elif document.get('schema') == 'test10.pair_selection.v1':
            items = document['pairs']
        else:
            raise ValueError('expected an LVOS raw_index.json or test10.pair_selection.v1')
        videos = {'train':set(), 'validation':set()}
        seen = set()
        for item in items:
            check_time(self.deadline)
            row = dict(item)
            if row['dataset'] != 'LVOSv2' or row['split'] not in videos:
                raise ValueError('this comparison recipe requires LVOSv2 train/validation only')
            file = (self.path.parent / row['path']).resolve()
            if pair_root:
                split = 'fit' if row['split'] == 'train' else 'development'
                file = Path(pair_root).resolve() / split / Path(row['path']).name
            if str(file) in seen or row.get('sha256') in {r['sha256'] for r in self.rows}:
                raise ValueError('duplicate pair path or content in selection')
            seen.add(str(file))
            row['path'] = str(file)
            row['stamp'] = list(signature(file))
            if not row.get('sha256'):
                row['sha256'] = file.with_suffix('.pt.sha256').read_text().split()[0]
            source, target, meta = self.load(row)
            indices = source.validity.nonzero().tolist()
            if 'valid_indices' in row and row['valid_indices'] != indices:
                raise ValueError('raw index validity differs from pair')
            if 'valid_records' in row and row['valid_records'] != len(indices):
                raise ValueError('raw index record count differs from pair')
            row.update(valid_indices=indices, valid_records=len(indices))
            videos[row['split']].add(row['video_id'])
            self.rows.append(row)
        if not all(videos.values()) or videos['train'] & videos['validation']:
            raise ValueError('nonempty video-disjoint fit/development splits required')
        if sha(self.path) != input_sha:
            raise ValueError('input manifest changed while reading pairs')
        self.collection = dict(schema=SCHEMA, selection=str(self.path), selection_sha256=input_sha,
                               source_index_sha256=input_sha if self.index else None,
                               fingerprint=data_fingerprint(self.rows), pairs=self.rows,
                               counts={split:dict(pairs=sum(r['split']==split for r in self.rows),
                                   videos=len(videos[split]), valid_records=sum(r['valid_records'] for r in self.rows if r['split']==split))
                                   for split in videos},
                               metadata_policy='original metadata retained; historical provenance may be unknown')

    def load(self, row):
        check_time(self.deadline)
        path = Path(row['path'])
        if list(signature(path)) != row['stamp']:
            raise ValueError('pair changed during training')
        checksum = path.with_suffix('.pt.sha256').read_text().split()[0]
        if checksum != row['sha256']:
            raise ValueError('pair sidecar/index checksum mismatch')
        if str(path) in self.cache:
            result, size = self.cache.pop(str(path))
            self.cache[str(path)] = result, size
            return result
        raw = path.read_bytes()
        import hashlib
        if hashlib.sha256(raw).hexdigest() != checksum or list(signature(path)) != row['stamp']:
            raise ValueError('pair bytes/checksum changed')
        with torch.serialization.safe_globals([CanonicalState, StateSpec]):
            payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        result = validate_pair(payload)
        if result[2]['num_maskmem'] != 7 or result[2]['max_obj_ptrs_in_encoder'] != 16:
            raise ValueError('comparison requires num_maskmem=7 and max_obj_ptrs_in_encoder=16')
        if result[2]['video_id'] != row['video_id']:
            raise ValueError('pair video differs from selection')
        if row.get('object_id') is not None and int(result[2]['object_id']) != int(row['object_id']):
            raise ValueError('pair object differs from raw index')
        if row.get('runtime_switch_frame') is not None and result[0].switch_frame != row['runtime_switch_frame']:
            raise ValueError('pair switch position differs from raw index')
        size = len(raw)
        if size <= self.cache_limit:
            while self.cache and self.cache_bytes + size > self.cache_limit:
                _, (_, old) = self.cache.popitem(last=False); self.cache_bytes -= old
            self.cache[str(path)] = result, size; self.cache_bytes += size
        return result

    def batches(self, split, size, epoch=None, seed=7):
        """Case order is shuffled with the reference generator; records keep their slot order
        and windows span case boundaries, exactly as lvos_ddp.schedule does."""
        rows = [r for r in self.rows if r['split'] == split]
        if epoch is not None:
            order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
            rows = [rows[i] for i in order]
        pending = []
        for row in rows:
            source, target, _ = self.load(row)
            for b, o, k in row['valid_indices']:
                pending.append((source.spatial_memory[b,o,k], source.object_pointer[b,o,k],
                                target.spatial_memory[b,o,k], target.object_pointer[b,o,k]))
                if len(pending) == size:
                    yield tuple(torch.stack(v) for v in zip(*pending)); pending = []
        if pending:
            yield tuple(torch.stack(v) for v in zip(*pending))


def check_time(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('time budget exhausted; incomplete training cannot be a completed comparison')


def fit_scales(data, deadline=None):
    """Fit-target RMS in FP64. With a raw index, train on its recorded scales as the reference does."""
    totals = {'spatial':[0.,0], 'pointer':[0.,0]}
    for batch in data.batches('train', 16):
        check_time(deadline)
        for name, value in zip(totals, batch[2:]):
            value = value.double()
            totals[name][0] += float(value.square().sum())
            totals[name][1] += value.numel()
    scales = {k:math.sqrt(s/n) for k, (s,n) in totals.items()}
    if data.index:
        recorded = data.index['normalization']
        if recorded['scope'] != 'fit_only' or any(not math.isclose(scales[k], recorded['scales'][k], rel_tol=1e-7, abs_tol=1e-12) for k in scales):
            raise ValueError('fit RMS does not match the raw index')
        return {k: float(recorded['scales'][k]) for k in scales}
    return scales


def objective(model, batch, scales, config, device):
    """lvos_training.component_objective: per-record mean, record mean, then RMS² weighting."""
    x,p,y,q = [v.to(device=device, dtype=torch.float32) for v in batch]
    spatial, pointer = model.translate_tensors(x[None,None], p[None,None])
    spatial_mse = (spatial[0,0]-y).square().flatten(1).mean(1).mean()
    pointer_mse = (pointer[0,0]-q).square().mean(1).mean()
    total = (config['lambda_spatial']*(spatial_mse/max(scales['spatial']**2,1e-12)) +
             config['lambda_pointer']*(pointer_mse/max(scales['pointer']**2,1e-12)))
    if not torch.isfinite(total): raise ValueError('nonfinite loss')
    return total, spatial_mse, pointer_mse


def optimizer_and_scheduler(model, config, updates):
    decay, no_decay = [], []
    norm = {id(p) for layer in model.modules() if isinstance(layer, torch.nn.LayerNorm) for p in layer.parameters(recurse=False)}
    for name, p in model.named_parameters():
        excluded = name.endswith('.bias') or id(p) in norm or name.endswith('.alpha') or 'pos_embed' in name
        (no_decay if excluded else decay).append(p)
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=config['weight_decay']),
                                   dict(params=no_decay, weight_decay=config['bias_weight_decay'])],
                                  lr=config['learning_rate'], betas=tuple(config['betas']), eps=config['eps'])
    warmup = max(1, math.ceil(updates * config['warmup_fraction']))
    def multiplier(step):
        if step < warmup: return (step+1)/warmup
        progress = min(1., (step-warmup)/max(1, updates-warmup))
        floor = config['min_learning_rate']/config['learning_rate']
        return floor + (1-floor)*.5*(1+math.cos(math.pi*progress))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def optimize_window(model, batch, optimizer, scales, config, device, deadline=None):
    """One global window: sum of loss×records over microbatches, then divide by the window's
    actual valid record count (lvos_ddp with world size 1)."""
    optimizer.zero_grad(set_to_none=True)
    count = len(batch[0]); totals = [0.,0.,0.]
    for start in range(0, count, config['microbatch_records']):
        check_time(deadline)
        tensors = tuple(v[start:start+config['microbatch_records']] for v in batch)
        records = len(tensors[0])
        losses = objective(model, tensors, scales, config, device)
        (losses[0]*records).backward()
        for i, loss in enumerate(losses): totals[i] += float(loss.detach())*records
    check_time(deadline)
    for parameter in model.parameters():
        if parameter.grad is not None: parameter.grad.mul_(1/count)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'], error_if_nonfinite=True)
    optimizer.step()
    return [total/count for total in totals]


def validate(model, data, scales, config, device, deadline=None):
    model.eval(); totals = [0.,0.,0.]; count = 0
    with torch.no_grad():
        for batch in data.batches('validation', config['microbatch_records']):
            check_time(deadline)
            for i, loss in enumerate(objective(model, batch, scales, config, device)):
                totals[i] += float(loss)*len(batch[0])
            count += len(batch[0])
    check_time(deadline)
    return dict(loss=totals[0]/count, spatial_mse=totals[1]/count, pointer_mse=totals[2]/count, records=count)


def affine_contract(model):
    """The comparison defines Affine by its function, not by a source revision of the translator repo."""
    names = [(n, tuple(p.shape)) for n, p in model.named_parameters()]
    if names != AFFINE_PARAMETERS:
        raise ValueError(f'Affine must be position-shared 64x64 spatial + 256x256 pointer Wx+b, got {names}')
    if getattr(model, 'output_dtype', None) != 'source' or model.output_dtypes(torch.bfloat16, torch.float32) != (torch.bfloat16, torch.float32):
        raise ValueError('Affine handoff must keep bf16 spatial memory and fp32 pointers')
    probe = copy.deepcopy(model).cpu().float()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in probe.parameters():
            parameter.copy_(.1*torch.randn(parameter.shape, generator=generator))
        x = torch.randn(1, 1, 3, 64, 64, 64, generator=generator); p = torch.randn(1, 1, 3, 256, generator=generator)
        spatial, pointer = probe.translate_tensors(x, p)
        expected = torch.nn.functional.linear(x.movedim(3, -1), probe.feature.weight, probe.feature.bias).movedim(-1, 3)
        torch.testing.assert_close(spatial, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(pointer, torch.nn.functional.linear(p, probe.pointer.weight, probe.pointer.bias), rtol=1e-5, atol=1e-6)
    try:
        revision = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        revision = revision.stdout.strip() if revision.returncode == 0 else 'unavailable'
    except FileNotFoundError:
        revision = 'unavailable'
    return dict(definition='position-shared spatial 64->64 and pointer 256->256 Wx+b; bf16 spatial handoff',
                parameters=sum(p.numel() for p in model.parameters()), class_name=type(model).__name__,
                translator_repository=str(REPO), translator_revision=revision)


def save_weights(path, method, model, data, epoch, steps, seed):
    payload = dict(schema=CHECKPOINT_SCHEMA, method=method, preset=PRESETS[method], epoch=epoch,
                   optimizer_steps=steps, seed=seed, training_fingerprint=data.collection['fingerprint'],
                   state_pair_schema=SCHEMA)
    if method == 'transformer': payload['model_payload'] = model.to_payload()
    else: payload['state_dict'] = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial'); torch.save(payload, temp); temp.replace(path)


def transformer_reference(run_dir, collection, scales, training_records, development_records, reference_index=None):
    """Read-only binding to the completed Transformer run: data, recipe, normalization and coverage.

    Reads run.json and metrics/history.json only; Transformer weights are never loaded."""
    cfg = recipe(); reference = Path(run_dir).resolve()
    evidence = snapshot([reference/'run.json', reference/'metrics/history.json'])
    run = read(reference/'run.json'); history = read(reference/'metrics/history.json')
    identity = run['identity']; config = identity['configuration']
    if identity['collection_digest'] != collection.get('source_index_sha256'):
        # A relocated/rebuilt index may differ byte-for-byte while selecting the same data.
        index = Path(reference_index) if reference_index else reference/'raw_index.json'
        if not index.is_file() or sha(index) != identity['collection_digest']:
            raise ValueError('provide --reference-index matching the Transformer run to compare data membership')
        evidence.update(snapshot([index]))
        source = read(index)
        if source.get('schema_version') != INDEX_SCHEMA: raise ValueError('unsupported reference index')
        if data_fingerprint(index_rows(source)) != collection['fingerprint']:
            raise ValueError('Transformer and Affine fit/development pairs or valid records differ')
    expected = dict(lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'], lambda_spatial=cfg['lambda_spatial'],
                    lambda_pointer=cfg['lambda_pointer'], lambda_cos=cfg['lambda_cos'], max_epochs=cfg['epochs'],
                    warmup_fraction=cfg['warmup_fraction'], min_lr=cfg['min_learning_rate'], clip_norm=cfg['clip_norm'],
                    seed=cfg['seed'], precision='fp32', augmentation='none')
    differences = {k:dict(expected=v, actual=config.get(k)) for k,v in expected.items() if config.get(k)!=v}
    if identity['training_contract']['global_batch'] != cfg['global_records']:
        differences['global_batch'] = identity['training_contract']['global_batch']
    if differences: raise ValueError(f'Transformer training recipe differs: {differences}')
    if identity.get('scope') != 'state_supervised_ddp' or run.get('JF_early_stopping_applied'):
        raise ValueError('reference must be the state-supervised DDP run without J&F early stopping')
    if sorted(h['epoch'] for h in history) != list(range(1,cfg['epochs']+1)):
        raise ValueError('Transformer needs all 30 completed epochs for this comparison')
    recorded = run['normalization']
    if recorded['scope'] != 'fit_only' or any(not math.isclose(recorded['scales'][k], v, rel_tol=1e-7, abs_tol=1e-12)
                                            for k,v in scales.items()):
        raise ValueError('Transformer/Affine normalization mismatch')
    per_epoch = math.ceil(training_records/cfg['global_records'])
    for h in history:
        if (h['records'] != training_records or h['dev']['records'] != development_records or
                h['optimizer_step'] != h['epoch']*per_epoch or not math.isfinite(h['dev']['loss'])):
            raise ValueError('Transformer training/development coverage or updates mismatch')
    return dict(run=str(reference), files=evidence, history=history,
                best_state_loss_epoch=select_epoch([dict(epoch=h['epoch'], loss=h['dev']['loss']) for h in history]),
                verified_conditions='same data membership, fit RMS, optimization recipe, epochs, coverage and updates',
                gpu_count_required_to_match=False, microbatch_required_to_match=False, weights_read=False)


def select_epoch(rows):
    """Reference best_state_loss.ckpt.json: rewritten whenever dev loss equals the running
    minimum, so an exact tie selects the later epoch."""
    best = None
    for row in sorted(rows, key=lambda r: r['epoch']):
        if best is None or row['loss'] <= best['loss']: best = row
    return best['epoch']


def train(args):
    started = time.monotonic(); config = recipe()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.budget_hours is not None and (not math.isfinite(args.budget_hours) or args.budget_hours <= 0):
        raise ValueError('budget-hours must be finite and positive')
    deadline = started + args.budget_hours*3600 if args.budget_hours is not None else None
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('use an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema=REPORT_SCHEMA, status='training', comparison_protocol=config['schema'],
                  requested_methods=[args.method], state_pair_schema=SCHEMA, config=config,
                  models={}, completed_epochs=0, seed=config['seed'])
    write(output/'train_report.json', report)
    try:
        data = PairData(args.pairs or args.index, args.pair_root, args.cache_gib, deadline)
        check_time(deadline)
        device = torch.device(args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
        import numpy
        torch.manual_seed(config['seed']); random.seed(config['seed']); numpy.random.seed(config['seed'])
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(config['seed'])
        torch.use_deterministic_algorithms(True)
        model = fresh_model(args.method).to(device=device, dtype=torch.float32)
        if args.method == 'affine': report['model_contract'] = affine_contract(model)
        count = sum(r['valid_records'] for r in data.rows if r['split']=='train')
        dev_count = sum(r['valid_records'] for r in data.rows if r['split']=='validation')
        updates = math.ceil(count/config['global_records'])*config['epochs']
        optimizer, scheduler = optimizer_and_scheduler(model, config, updates)
        scales = fit_scales(data, deadline)
        reference = None
        if getattr(args, 'transformer_run', None):
            reference = transformer_reference(args.transformer_run, data.collection, scales, count, dev_count,
                                              getattr(args, 'reference_index', None))
        files = snapshot([data.path, RECIPE_PATH, ROOT/'comparable_training.py', ROOT/'training.py', ROOT/'state_pairs.py',
                          ROOT/'core.py', ROOT/'manifest.py', *sorted((REPO/'src').rglob('*.py'))])
        report.update(collection=data.collection, files=files,
                      normalization=dict(scope='fit_only', scales=scales,
                                         source='raw_index' if data.index else 'recomputed_fit_targets'),
                      device=str(device), planned_updates=updates, training_records=count,
                      development_records=dev_count, torch=str(torch.__version__), history=[],
                      optimizer_steps=0, epoch_checkpoints=[],
                      transformer_reference={k:v for k,v in reference.items() if k != 'history'} if reference else None)
        write(output/'training_inputs.json', report)
        steps = 0
        for epoch in range(config['epochs']):
            check_time(deadline); model.train(); totals=[0.,0.,0.]; seen=0
            for batch in data.batches('train', config['global_records'], epoch, config['seed']):
                losses = optimize_window(model, batch, optimizer, scales, config, device, deadline)
                scheduler.step(); steps += 1
                for i, loss in enumerate(losses): totals[i] += loss*len(batch[0])
                seen += len(batch[0])
                report['optimizer_steps'] = steps
            if seen != count: raise ValueError('incomplete epoch record coverage')
            dev = validate(model, data, scales, config, device, deadline)
            row = dict(epoch=epoch+1, optimizer_steps=steps, train_loss=totals[0]/seen,
                       training_records=seen, validation=dev, learning_rate=optimizer.param_groups[0]['lr'])
            report['history'].append(row); report['completed_epochs']=epoch+1
            path = output/'epochs'/f'{args.method}-epoch-{epoch+1:05d}.pt'
            save_weights(path, args.method, model, data, epoch+1, steps, config['seed'])
            report['epoch_checkpoints'].append(dict(epoch=epoch+1, checkpoint=str(path.relative_to(output)), sha256=sha(path)))
            if epoch+1 == select_epoch([dict(epoch=h['epoch'], loss=h['validation']['loss']) for h in report['history']]):
                target = output/f'{args.method}.pt'
                shutil.copyfile(path, target.with_suffix('.partial')); target.with_suffix('.partial').replace(target)
                report['models'][args.method] = dict(checkpoint=target.name, sha256=sha(target), preset=PRESETS[args.method],
                    epoch=epoch+1, optimizer_steps=steps, origin='fresh', validation=dev,
                    selection=dict(rule=config['checkpoint_selection'], epoch=epoch+1),
                    parameters=sum(p.numel() for p in model.parameters()))
            write(output/f'{args.method}.history.json', report['history'])
            write(output/'train_report.json', report)
            print(f"{args.method}: epoch {epoch+1}/{config['epochs']}, records={seen}, dev={dev['loss']:.6g}", flush=True)
        verify_snapshot(files)
        if reference: verify_snapshot(reference['files'])
        report.update(status='complete', stop_reason='all_epochs_completed')
    except TimeoutError:
        report.update(status='incomplete', stop_reason='time_budget')
    except BaseException as exc:
        report.update(status='failed', stop_reason=type(exc).__name__, error=str(exc))
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        write(output/'train_report.json', report)
    return report


def select_checkpoint(args):
    """New training directory whose Affine checkpoint is a declared epoch of a completed run.

    Use this only to apply the rule that chose the Transformer checkpoint being compared
    (for example its final epoch); never choose by test10 evaluation scores."""
    source = Path(args.affine_dir).resolve()
    fit = training_report(source, ['affine'])
    if fit.get('comparison_protocol') != recipe()['schema'] or fit['config'] != recipe():
        raise ValueError('Affine was not trained with the fixed comparison recipe')
    if not args.reason.strip(): raise ValueError('record why this epoch is the comparison checkpoint')
    entry = next((e for e in fit.get('epoch_checkpoints', []) if e['epoch'] == args.epoch), None)
    row = next((h for h in fit['history'] if h['epoch'] == args.epoch), None)
    if entry is None or row is None: raise ValueError(f'epoch {args.epoch} has no saved Affine checkpoint')
    if sha(source/entry['checkpoint']) != entry['sha256']: raise ValueError('epoch checkpoint checksum mismatch')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('use a new empty directory')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source/entry['checkpoint'], output/'affine.pt')
    report = copy.deepcopy(fit)
    report['models']['affine'] = dict(fit['models']['affine'], checkpoint='affine.pt', sha256=sha(output/'affine.pt'),
        epoch=args.epoch, optimizer_steps=row['optimizer_steps'], validation=row['validation'],
        selection=dict(rule='declared_epoch', epoch=args.epoch, reason=args.reason.strip(), source=str(source),
                       default_rule_epoch=fit['models']['affine']['epoch']))
    write(output/'train_report.json', report)
    return report


def attach_transformer(args):
    """Import the best full-development state-loss export; never use test10 GT for selection."""
    fit = training_report(args.affine_dir, ['affine'])
    cfg = recipe()
    if fit.get('comparison_protocol') != cfg['schema'] or fit['config'] != cfg:
        raise ValueError('Affine was not trained with the fixed comparison recipe')
    dev_records = sum(r['valid_records'] for r in fit['collection']['pairs'] if r['split']=='validation')
    bound = transformer_reference(args.transformer_run, fit['collection'], fit['normalization']['scales'],
                                  fit['training_records'], dev_records, getattr(args, 'reference_index', None))
    reference, evidence, history = Path(bound['run']), bound['files'], bound['history']
    identity = read(reference/'run.json')['identity']
    chosen = next(h for h in history if h['epoch'] == bound['best_state_loss_epoch'])
    marker = reference/'best_state_loss.ckpt.json'
    if marker.is_file() and f"epoch-{chosen['epoch']:05d}" not in read(marker)['path']:
        raise ValueError('best_state_loss.ckpt.json disagrees with the development history')
    export_path = reference/f"translator/epoch-{chosen['epoch']:05d}.pth"
    meta_path = export_path.with_suffix('.json')
    evidence.update(snapshot([meta_path, export_path]))
    meta = read(meta_path)
    if meta['identity'] != identity or meta['epoch'] != chosen['epoch'] or meta['weights']['sha256'] != sha(export_path):
        raise ValueError('Transformer export metadata/checksum mismatch')
    exported = torch.load(export_path, map_location='cpu', weights_only=True)
    model = TransformerStateTranslator.from_payload(exported)
    if exported.get('preset') != 'base': raise ValueError('expected the base Transformer export')
    if model.source_spec != SAM21_MEMORY_SPEC or model.target_spec != SAM21_MEMORY_SPEC:
        raise ValueError('Transformer export memory spec mismatch')
    verify_snapshot(evidence)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('use a new empty comparison directory')
    output.mkdir(parents=True, exist_ok=True)
    report = copy.deepcopy(fit)
    for row in fit['models'].values(): shutil.copy2(Path(args.affine_dir)/row['checkpoint'], output/row['checkpoint'])
    payload = dict(schema=CHECKPOINT_SCHEMA, method='transformer', preset='base', epoch=chosen['epoch'],
                   optimizer_steps=chosen['optimizer_step'], seed=cfg['seed'], state_pair_schema=SCHEMA,
                   training_fingerprint=fit['collection']['fingerprint'], model_payload=exported)
    path = output/'transformer.pt'; temp=path.with_suffix('.partial'); torch.save(payload,temp); temp.replace(path)
    report['models']['transformer'] = dict(checkpoint=path.name, sha256=sha(path), preset='base',
        epoch=chosen['epoch'], optimizer_steps=chosen['optimizer_step'], origin='verified_external',
        parameters=sum(p.numel() for p in model.parameters()), validation=chosen['dev'])
    report['requested_methods'] = ['affine','transformer']
    report['files'].update(evidence)
    report['transformer_reference'] = dict({k:v for k,v in bound.items() if k != 'history'}, files=evidence,
                                           selected_epoch=chosen['epoch'], weights_read=True)
    write(output/'train_report.json', report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train', help='fixed 30-epoch recipe; defaults to Affine only')
    inputs = t.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--index', type=Path, help='original LVOS raw_index.json')
    inputs.add_argument('--pairs', type=Path, help='explicit LVOS-only test10.pair_selection.v1')
    t.add_argument('--pair-root', type=Path, help='local root with fit/ and development/; preserve file names')
    t.add_argument('--method', choices=('affine','transformer'), default='affine')
    t.add_argument('--output-dir', required=True, type=Path)
    t.add_argument('--device', default='auto')
    t.add_argument('--cache-gib', type=float, default=2.)
    t.add_argument('--budget-hours', type=float, help='optional cap; incomplete epochs/runs are not completed comparisons')
    t.add_argument('--transformer-run', type=Path,
                   help='completed Transformer DDP run dir; checks run.json/history.json before training, reads no weights')
    t.add_argument('--reference-index', type=Path, help='original Transformer index if the Affine input manifest differs')
    s = sub.add_parser('select', help='copy a completed run with a declared Affine epoch as the comparison checkpoint')
    s.add_argument('--affine-dir', required=True, type=Path)
    s.add_argument('--epoch', required=True, type=int)
    s.add_argument('--reason', required=True, help='rule that chose the compared Transformer checkpoint')
    s.add_argument('--output-dir', required=True, type=Path)
    a = sub.add_parser('attach-transformer', help='bundle an already trained LVOS Transformer with Affine')
    a.add_argument('--affine-dir', required=True, type=Path)
    a.add_argument('--transformer-run', required=True, type=Path)
    a.add_argument('--reference-index', type=Path, help='original Transformer index if the Affine input manifest differs')
    a.add_argument('--output-dir', required=True, type=Path)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    result = {'train': train, 'select': select_checkpoint, 'attach-transformer': attach_transformer}[args.command](args)
    if result['status'] != 'complete': raise SystemExit('incomplete run; see train_report.json')
