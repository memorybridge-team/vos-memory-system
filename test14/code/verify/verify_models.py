#!/usr/bin/env python3
"""Verify the three state translators. Reads source/caches; writes only under VERIFY_DIR."""
import copy
import dataclasses
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from torch import nn

from vos_memory_inspector.models._spatial import SAM21_MEMORY_SPEC as SPEC
from vos_memory_inspector.models.registry import build_translator
from vos_memory_inspector.models.transformer import TransformerStateTranslator
from vos_memory_inspector.state_schema import CanonicalState
from vos_memory_inspector.state_training.data import load_index, load_pair, record_loader
from vos_memory_inspector.state_training.evaluation import evaluate_artifact
from vos_memory_inspector.state_training.policy import TrainConfig, optimizer_groups
from vos_memory_inspector.state_training.preflight import REFERENCE_INDEX_SHA256, inspect
from vos_memory_inspector.state_training.ridge import SCHEMA, fit_ridge, load_ridge
from vos_memory_inspector.state_training.storage import save_json
from vos_memory_inspector.state_training.trainer import construct_model, evaluate, restore_model, train
from vos_memory_inspector.translator_artifact import load_learned_translator

WORK = Path(os.environ['VERIFY_DIR'])
SRC = Path(os.environ['CMMT_PREP_ROOT']) / 'source'
BASE = Path('/workspace/j700qtcgrg/CMMT-official-isolated/official-20261005T012256KST')
OFFICIAL = BASE / 'index/raw_index.json'
LEGACY_DIR = BASE / 'final-state-loss-20261005T1627KST/delivery/selected_state_loss_best'
ROOT_MAP = {'/workspace/CMMT-task07-artifacts/production':
            '/workspace/j700qtcgrg/CMMT-task07-artifacts/production'}
MODELS = ('residual_mlp', 'transformer', 'mlp_context')
EXPECT = {
    'residual_mlp': dict(cls='ResidualMLPStateTranslator', spatial=20736, pointer=328704, total=349440,
                         readme_spatial=16576, readme_total=279488, readme_macs=67.1e6),
    'transformer': dict(cls='TransformerStateTranslator', spatial=194304, pointer=262912, total=457216,
                        readme_spatial=194304, readme_total=457216, readme_macs=77.6e6),
    'mlp_context': dict(cls='MLPContextStateTranslator', spatial=210880, pointer=328704, total=539584,
                        readme_spatial=206720, readme_total=469632, readme_macs=127.9e6),
}
RESULTS, INFO = [], {}
DEVICE = None


def check(group, name, ok, detail=''):
    RESULTS.append(dict(group=group, name=name, ok=bool(ok), detail=str(detail)[:1500]))
    print(f"[{'PASS' if ok else 'FAIL'}] {group} | {name} | {str(detail)[:400]}", flush=True)


def guarded(function, *args):
    started = time.monotonic()
    print(f'\n===== {function.__name__} =====', flush=True)
    try:
        value = function(*args)
    except Exception:
        check(function.__name__, 'section completed', False, traceback.format_exc()[-1400:])
        value = None
    print(f'----- {function.__name__}: {time.monotonic() - started:.1f}s', flush=True)
    return value


def raw_config(name):
    raw = json.loads((SRC / 'configs/state_training' / f'{name}.json').read_text())
    if raw['model'] != name:
        raise ValueError('config/model mismatch')
    return raw


def build(name, seed=7):
    torch.manual_seed(seed)
    return construct_model(name, SPEC, raw_config(name)['model_options'])


def fake_ridge():
    g = torch.Generator().manual_seed(1)

    def affine(n):
        return dict(weight=torch.eye(n) * .9 + .02 * torch.randn(n, n, generator=g),
                    bias=.1 * torch.randn(n, generator=g))
    return dict(schema_version=SCHEMA, source_spec=SPEC.to_dict(), target_spec=SPEC.to_dict(),
                spatial=affine(64), pointer=affine(256),
                provenance=dict(artifact_sha256='0' * 64, data_identity={'synthetic': True}))


def activate(model):
    """Give zero-initialized parts values so every branch contributes."""
    with torch.no_grad():
        for parameter in model.parameters():
            if bool((parameter == 0).all()):
                nn.init.normal_(parameter, std=.05)
    return model


def conv(layer, a, b, k=1):
    return isinstance(layer, nn.Conv2d) and (layer.in_channels, layer.out_channels, layer.kernel_size[0]) == (a, b, k)


def linear(layer, a, b):
    return isinstance(layer, nn.Linear) and (layer.in_features, layer.out_features) == (a, b)


def structure():
    for name, exp in EXPECT.items():
        model = build(name)
        counts = model.parameter_breakdown()
        check('구조', f'{name} 클래스', type(model).__name__ == exp['cls'], type(model).__name__)
        check('구조', f'{name} 파라미터(설정 파일 기준)',
              (counts['spatial'], counts['pointer'], counts['total']) == (exp['spatial'], exp['pointer'], exp['total']),
              counts)
        check('구조', f'{name} 공간+포인터=전체', counts['spatial'] + counts['pointer'] == counts['total'], counts)
        options = dict(raw_config(name)['model_options'])
        if name != 'transformer':
            options['skip_mode'] = 'identity'
        plain = construct_model(name, SPEC, options)
        plain_counts = plain.parameter_breakdown()
        check('구조', f'{name} README 표는 identity 기준과 일치',
              (plain_counts['spatial'], plain_counts['total']) == (exp['readme_spatial'], exp['readme_total']), plain_counts)
        check('구조', f'{name} README MAC은 identity 기준과 일치',
              abs(plain.macs_per_frame() - exp['readme_macs']) < 0.06e6, plain.macs_per_frame())
        INFO.setdefault('structure', {})[name] = dict(
            config_parameters=counts, identity_parameters=plain_counts, macs_per_frame=model.macs_per_frame(),
            identity_macs_per_frame=plain.macs_per_frame(), pointer_macs=model.pointer_macs_per_record(),
            skip_mode=getattr(model, 'skip_mode', None),
            leaf_modules=[f'{n}: {m}' for n, m in model.named_modules() if not list(m.children())])
    m = build('residual_mlp')
    check('구조', 'residual_mlp 층 구성', linear(m.feature[0], 64, 128) and isinstance(m.feature[1], nn.GELU)
          and linear(m.feature[2], 128, 64) and linear(m.pointer[0], 256, 512) and isinstance(m.pointer[1], nn.GELU)
          and linear(m.pointer[2], 512, 256) and linear(m.feature_skip, 64, 64) and linear(m.pointer_skip, 256, 256))
    check('구조', 'residual_mlp 문맥 없음(Conv/Attention 없음)',
          not any(isinstance(x, nn.Conv2d) or 'Attention' in type(x).__name__ for x in m.modules()))
    m = build('transformer')
    c = m.config
    check('구조', 'transformer 설정', (c.local_fusion, c.skip_mode, c.patch_size, c.num_tokens, c.d_model, c.num_layers,
          c.num_heads, c.pos_embed, c.pointer_hidden_dim) == ('linear', 'identity', 4, 256, 64, 2, 4, 'learnable', 512), c)
    check('구조', 'transformer 층 구성', conv(m.spatial.local_in, 64, 64) and m.spatial.local_out is None
          and m.spatial.linear_skip is None and m.pointer.linear_skip is None and len(m.spatial.blocks) == 2
          and conv(m.spatial.context_conv, 64, 64, 3) and conv(m.spatial.context_proj, 64, 64))
    m = build('mlp_context')
    c = m.config
    check('구조', 'mlp_context 설정', (c.local_fusion, c.skip_mode, c.fusion_hidden_dim, c.patch_size, c.num_layers,
          c.num_heads, c.pointer_hidden_dim) == ('additive_mlp', 'affine', 128, 4, 2, 4, 512), c)
    check('구조', 'mlp_context 층 구성', conv(m.spatial.local_in, 64, 128) and conv(m.spatial.local_out, 128, 64)
          and linear(m.spatial.linear_skip, 64, 64) and linear(m.pointer.linear_skip, 256, 256)
          and len(m.spatial.blocks) == 2 and isinstance(m, TransformerStateTranslator))
    INFO['registry_defaults'] = {name: getattr(build_translator(p, SPEC, SPEC), 'skip_mode', None)
                                 for name, p in (('residual_mlp', 'residual_mlp'), ('transformer', 'base'), ('mlp_context', 'mlp_context'))}
    print('registry default skip_mode', INFO['registry_defaults'], flush=True)
    for name in MODELS:
        a, b = build(name), build(name)
        check('구조', f'{name} 같은 seed면 같은 초기값',
              all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values())))


def init_forward():
    torch.manual_seed(3)
    x = torch.randn(1, 1, 6, 64, 64, 64).bfloat16().to(DEVICE)
    p = torch.randn(1, 1, 6, 256).to(DEVICE)
    valid = torch.ones(1, 1, 6, dtype=torch.bool, device=DEVICE)
    valid[0, 0, 4] = False
    zero_parts = dict(residual_mlp=('feature.2.weight', 'feature.2.bias', 'pointer.2.weight', 'pointer.2.bias'),
                      transformer=('spatial.alpha', 'pointer.fc2.weight', 'pointer.fc2.bias'),
                      mlp_context=('spatial.alpha', 'spatial.local_out.weight', 'spatial.local_out.bias',
                                   'pointer.fc2.weight', 'pointer.fc2.bias'))
    for name in MODELS:
        model = build(name).to(DEVICE)
        state = dict(model.named_parameters())
        check('초기화', f'{name} 0 초기화 대상', all(bool((state[k] == 0).all()) for k in zero_parts[name]), zero_parts[name])
        s, q = model.translate_tensors(x, p, valid)
        check('형상', f'{name} 학습 API 출력', s.shape == x.shape and q.shape == p.shape and s.dtype == q.dtype == torch.float32,
              (tuple(s.shape), tuple(q.shape), s.dtype))
        delta = max(float((s[valid] - x[valid].float()).abs().max()), float((q[valid] - p[valid]).abs().max()))
        check('초기화', f'{name} ridge 전 출력=입력', delta == 0, delta)
        s, q = model.translate_handoff_tensors(x, p, valid)
        check('형상', f'{name} handoff dtype', s.dtype == torch.bfloat16 and q.dtype == torch.float32, (s.dtype, q.dtype))
        check('패딩', f'{name} 패딩 슬롯은 원본 그대로', torch.equal(s[0, 0, 4], x[0, 0, 4]) and torch.equal(q[0, 0, 4], p[0, 0, 4]))
        active = activate(copy.deepcopy(model))
        base_s, base_q = active.translate_tensors(x, p, valid)
        x_pad = x.clone()
        x_pad[0, 0, 4] += 5
        s2, q2 = active.translate_tensors(x_pad, p, valid)
        check('패딩', f'{name} 패딩 값이 유효 기록에 영향 없음', torch.equal(s2[valid], base_s[valid]) and torch.equal(q2[valid], base_q[valid]))
        x_other = x.clone()
        x_other[0, 0, 0] += 1
        s3, _ = active.translate_tensors(x_other, p, valid)
        check('독립성', f'{name} 기록 간 섞임 없음', torch.equal(s3[0, 0, 1:4], base_s[0, 0, 1:4]))
        check('독립성', f'{name} 변경한 기록은 출력 변화', not torch.equal(s3[0, 0, 0], base_s[0, 0, 0]))
        x_pixel = x.clone()
        x_pixel[0, 0, 1, :, 10, 10] += 1
        s4, _ = active.translate_tensors(x_pixel, p, valid)
        changed = int(((s4 - base_s)[0, 0, 1].abs().amax(0) > 1e-6).sum())
        expected = changed == 1 if name == 'residual_mlp' else changed > 1
        check('공간 문맥', f'{name} 한 픽셀 변경의 영향 범위', expected, f'변한 위치 {changed}/4096')
        wrong = torch.zeros(1, 1, 1, 32, 64, 64, device=DEVICE)
        try:
            model.translate_tensors(wrong, torch.zeros(1, 1, 1, 256, device=DEVICE))
            rejected = False
        except ValueError:
            rejected = True
        check('형상', f'{name} 잘못된 입력 거부', rejected)
    ridge = fake_ridge()
    for name in ('residual_mlp', 'mlp_context'):
        model = build(name).to(DEVICE)
        try:
            model.to_payload()
            refused = False
        except ValueError:
            refused = True
        check('초기화', f'{name} ridge 없이 저장 거부', refused)
        model.cpu().initialize_from_ridge(ridge)
        model.to(DEVICE)
        s, q = model.translate_tensors(x, p, valid)
        ws, bs = ridge['spatial']['weight'].to(DEVICE), ridge['spatial']['bias'].to(DEVICE)
        wp, bp = ridge['pointer']['weight'].to(DEVICE), ridge['pointer']['bias'].to(DEVICE)
        want_s = (x.float().movedim(3, -1) @ ws + bs).movedim(-1, 3)
        want_q = p @ wp + bp
        delta = max(float((s[valid] - want_s[valid]).abs().max()), float((q[valid] - want_q[valid]).abs().max()))
        check('초기화', f'{name} ridge 후 출력=행벡터 ridge 해', delta < 1e-5, delta)
    try:
        construct_model('transformer', SPEC, dict(skip_mode='affine'))
        refused = False
    except ValueError:
        refused = True
    check('초기화', 'transformer는 affine skip 거부(기존 base 유지)', refused)


def roundtrip():
    ridge = fake_ridge()
    schema = dict(residual_mlp='cmmt.residual_mlp_translator.v3',
                  transformer='cmmt.spatial_context_transformer_translator.v1',
                  mlp_context='cmmt.spatial_context_transformer_translator.v2')
    torch.manual_seed(5)
    x, p = torch.randn(1, 1, 3, 64, 64, 64), torch.randn(1, 1, 3, 256)
    for name in MODELS:
        model = build(name)
        if getattr(model, 'skip_mode', 'identity') == 'affine':
            model.initialize_from_ridge(ridge)
        activate(model).eval()
        buffer = io.BytesIO()
        torch.save(model.to_payload(), buffer)
        buffer.seek(0)
        payload = torch.load(buffer, map_location='cpu', weights_only=True)
        check('저장', f'{name} payload schema', payload['schema_version'] == schema[name], payload['schema_version'])
        restored = restore_model(name, payload)
        check('저장', f'{name} 복원 클래스', type(restored) is type(model), type(restored).__name__)
        with torch.no_grad():
            a, b = model.translate_tensors(x, p), restored.translate_tensors(x, p)
        check('저장', f'{name} 복원 후 출력 동일', torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))


def weight_decay_groups():
    for name in MODELS:
        model = build(name)
        groups = optimizer_groups(model, 1e-4)
        names = {id(p): n for n, p in model.named_parameters()}
        decay = [names[id(p)] for p in groups[0]['params']]
        excluded = [names[id(p)] for p in groups[1]['params']]
        norms = {f'{n}.{k}' for n, m in model.named_modules() if isinstance(m, nn.LayerNorm) for k, _ in m.named_parameters(recurse=False)}
        check('옵티마이저', f'{name} 모든 파라미터를 한 번씩 포함',
              sorted(decay + excluded) == sorted(names.values()) and not set(decay) & set(excluded), (len(decay), len(excluded)))
        check('옵티마이저', f'{name} bias·LN·gate·위치 임베딩은 decay 제외',
              all(n.endswith('.bias') or n in norms or n.endswith('.alpha') or 'pos_embed' in n for n in excluded)
              and not any(n.endswith('.bias') or n in norms or n.endswith('.alpha') or 'pos_embed' in n for n in decay))
        skips = [n for n in names.values() if 'skip' in n and n.endswith('.weight')]
        check('옵티마이저', f'{name} affine skip weight는 decay 적용', all(n in decay for n in skips), skips)


def prepare_real():
    raw = json.loads(OFFICIAL.read_text())
    official = load_index(OFFICIAL, ROOT_MAP)
    check('실제 데이터', '공식 index 로드', official.official and official.identity['cases'] == 23358, official.identity)
    check('실제 데이터', '공식 index SHA가 1차 학습과 같음', official.identity['index_sha256'] == REFERENCE_INDEX_SHA256)
    groups = {}
    for entry in raw['entries']:
        case = entry['case']
        groups.setdefault((case['dataset'], case['training_role'], case.get('origin_paired_split')), []).append(entry)
    chosen, seen = [], set()
    for key, count in ((('MOSEv2', 'train', 'fit'), 2), (('MOSEv2', 'train', 'development'), 2),
                       (('LVOS v2', 'train', 'fit'), 2), (('LVOS v2', 'train', 'development'), 2),
                       (('LVOS v2', 'validation', 'validation'), 4)):
        picked = 0
        for entry in groups[key]:
            video = (entry['case']['dataset'], entry['case']['video_id'])
            if video in seen or entry['records'] > 24:
                continue
            seen.add(video)
            chosen.append(entry)
            picked += 1
            if picked == count:
                break
    path = WORK / 'sample-index.json'
    save_json(path, dict(schema_version='cmmt.state_training_sample_index.v1', entries=chosen))
    index = load_index(path, ROOT_MAP, allow_sample=True)
    training = dict(raw_config('transformer')['training'])
    config = TrainConfig(**{**training, 'workers': 0, 'pin_memory': False, 'max_epochs': 2, 'min_epochs': 1})
    config.validate()
    report = inspect(index, config)
    check('실제 데이터', '샘플 pair 체크섬·텐서 계약', report['sample_data_ready'] and not report['invalid'] and not report['missing'],
          dict(cases=report['checked_cases'], records=report['checked_records'], invalid=report['invalid'][:2]))
    check('실제 데이터', '실제 spec이 모델 spec과 같음', report['source_spec'] == SPEC.to_dict(), report['source_spec'])
    value, _ = load_pair(index.entries[0])
    s, t = value['source_canonical'], value['target_canonical']
    INFO['pair'] = dict(direction=(value['metadata'].get('source_model_id'), value['metadata'].get('target_model_id')),
                        spatial=list(s.spatial_memory.shape), pointer=list(s.object_pointer.shape),
                        spatial_dtype=str(s.spatial_memory.dtype), pointer_dtype=str(s.object_pointer.dtype),
                        sample_cases=[e['case']['case_id'] for e in index.entries])
    print('pair', INFO['pair'], flush=True)
    check('실제 데이터', 'source·target spec 동일', s.spec == t.spec == SPEC)
    ridge = fit_ridge(index, config, WORK / 'ridge.pth')
    INFO['ridge_alpha'] = {c: [x['alpha'] for x in ridge['candidates'][c] if x['selected']] for c in ('spatial', 'pointer')}
    print('selected ridge alpha', INFO['ridge_alpha'], flush=True)
    check('실제 데이터', 'ridge 데이터 identity 일치', ridge['provenance']['data_identity'] == index.identity)
    return dict(official=official, index=index, config=config, report=report, ridge=ridge,
                scales=official.normalization['scales'])


def objective(s, p, batch, scales):
    spatial = (s - batch['target_spatial'].float()).square().flatten(1).mean(1).mean() / scales['spatial'] ** 2
    pointer = (p - batch['target_pointer'].float()).square().mean(1).mean() / scales['pointer'] ** 2
    return spatial + pointer


def real_device(ctx):
    index, config, ridge, scales = ctx['index'], ctx['config'], ctx['ridge'], ctx['scales']
    loader = record_loader(index.role('train'), dataclasses.replace(config, microbatch=64))
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in next(iter(loader)).items()}
    count = batch['source_spatial'].shape[0]
    x, y = batch['source_spatial'], batch['source_pointer']
    direct = float(objective(x.float(), y, batch, scales))
    ws, bs = ridge['spatial']['weight'].to(DEVICE), ridge['spatial']['bias'].to(DEVICE)
    wp, bp = ridge['pointer']['weight'].to(DEVICE), ridge['pointer']['bias'].to(DEVICE)
    ridge_s = (x.float().movedim(1, -1) @ ws + bs).movedim(-1, 1)
    ridge_p = y @ wp + bp
    ridged = float(objective(ridge_s, ridge_p, batch, scales))
    INFO['real_reference'] = dict(records=count, direct_copy=direct, sample_ridge=ridged)
    print('reference', INFO['real_reference'], flush=True)
    check('실제 데이터', 'ridge 손실 < Direct Copy 손실', ridged < direct, (ridged, direct))
    for name in MODELS:
        model = build(name)
        affine = getattr(model, 'skip_mode', 'identity') == 'affine'
        if affine:
            model.initialize_from_ridge(ridge)
        model.to(device=DEVICE, dtype=torch.float32).train()

        def forward():
            s, p = model.translate_tensors(x[None, None], y[None, None])
            return s[0, 0], p[0, 0]
        s, p = forward()
        check('실제 데이터', f'{name} 출력 유한', bool(torch.isfinite(s).all() and torch.isfinite(p).all()))
        initial = float(objective(s, p, batch, scales))
        if affine:
            delta = max(float((s - ridge_s).abs().max()), float((p - ridge_p).abs().max()))
            check('실제 데이터', f'{name} 초기 출력=ridge', delta < 1e-3, delta)
        else:
            delta = max(float((s - x.float()).abs().max()), float((p - y).abs().max()))
            check('실제 데이터', f'{name} 초기 출력=Direct Copy', delta == 0, delta)
        optimizer = torch.optim.AdamW(optimizer_groups(model, config.weight_decay), lr=config.lr)

        def step():
            optimizer.zero_grad(set_to_none=True)
            a, b = forward()
            loss = objective(a, b, batch, scales)
            loss.backward()
            bad = [n for n, q in model.named_parameters() if q.grad is None or not bool(torch.isfinite(q.grad).all())]
            silent = [n for n, q in model.named_parameters() if q.grad is not None and float(q.grad.abs().max()) == 0]
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm, error_if_nonfinite=True)
            optimizer.step()
            return float(loss.detach()), bad, silent
        first_loss, bad, silent = step()
        check('기울기', f'{name} 첫 step 기울기 유한', not bad, bad[:5])
        INFO.setdefault('zero_gradient_first_step', {})[name] = silent
        print(f'{name} 첫 step 기울기 0 파라미터 {len(silent)}개: {silent}', flush=True)
        step()
        _, bad, silent = step()
        check('기울기', f'{name} 3번째 step에는 모든 파라미터에 기울기', not bad and not silent, (bad[:5], silent[:8]))
        trace = [first_loss]
        started = time.monotonic()
        for iteration in range(4, 101):
            value, _, _ = step()
            if iteration in (25, 50, 75, 100):
                trace.append(value)
        with torch.no_grad():
            final = float(objective(*forward(), batch, scales))
        check('학습 가능성', f'{name} 작은 배치 과적합으로 손실 감소', final < initial * 0.98,
              dict(initial=initial, trace=trace, final=final, ratio=final / initial, seconds=round(time.monotonic() - started, 1)))
        INFO.setdefault('overfit', {})[name] = dict(initial=initial, trace=trace, final=final, ratio=final / initial)
        if DEVICE.type == 'cuda':
            model_cpu = copy.deepcopy(model).cpu().eval()
            model.eval()
            with torch.no_grad():
                a, b = model.translate_tensors(x[:4][None, None], y[:4][None, None])
                c, d = model_cpu.translate_tensors(x[:4].cpu()[None, None], y[:4].cpu()[None, None])
            delta = max(float((a.cpu() - c).abs().max()), float((b.cpu() - d).abs().max()))
            check('장치', f'{name} CPU·GPU 출력 일치', delta < 1e-3, delta)
            model.train()
            fp_x, fp_y = x[:32], y[:32]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            for _ in range(10):
                optimizer.zero_grad(set_to_none=True)
                s, p = model.translate_tensors(fp_x[None, None], fp_y[None, None])
                (s.square().mean() + p.square().mean()).backward()
                optimizer.step()
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            records = fp_x.shape[0] * 10
            INFO.setdefault('gpu', {})[name] = dict(records_per_second=round(records / seconds, 1),
                                                    peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                                                    microbatch=int(fp_x.shape[0]))
            print(name, 'GPU', INFO['gpu'][name], flush=True)
        del model, optimizer
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()


def real_pipeline(ctx):
    index, config, report, ridge = ctx['index'], ctx['config'], ctx['report'], ctx['ridge']
    ridge_path = WORK / 'ridge.pth'
    ridge_eval = evaluate_artifact(index, config, 'ridge', ridge_path, WORK / 'eval-ridge.json', allow_sample=True, readiness=report)
    INFO['sample_validation'] = dict(ridge=ridge_eval['metrics']['all']['state_loss'])
    for name in MODELS:
        raw = raw_config(name)
        options = raw['model_options']
        affine = options.get('skip_mode', 'identity') == 'affine'
        output = WORK / 'runs' / name
        model, history = train(index, config, name, output, options=options, ridge_path=ridge_path if affine else None,
                               device='cpu', allow_sample=True, readiness=report)
        check('학습 경로', f'{name} train() 완료', len(history) == 2, [round(h['validation']['all']['state_loss'], 5) for h in history])
        for item in ('last.ckpt', 'best.ckpt', 'paired_translators.pt', 'training_report.json'):
            check('학습 경로', f'{name} {item} 생성', (output / item).is_file())
        artifact = output / 'paired_translators.pt'
        loaded = load_learned_translator(artifact, name)
        check('학습 경로', f'{name} handoff 로더 클래스', type(loaded).__name__ == EXPECT[name]['cls'], type(loaded).__name__)
        same = all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(loaded.state_dict().values(), model.state_dict().values()))
        check('학습 경로', f'{name} 저장 가중치=선택된 모델', same)
        evaluated = evaluate_artifact(index, config, name, artifact, output / 'evaluation.json', allow_sample=True, readiness=report)
        INFO['sample_validation'][name] = evaluated['metrics']['all']['state_loss']
        check('학습 경로', f'{name} evaluate 완료', evaluated['metrics']['all']['state_loss'] > 0, INFO['sample_validation'][name])
        resumed, more = train(index, dataclasses.replace(config, max_epochs=3), name, output, options=options,
                              ridge_path=ridge_path if affine else None, device='cpu', allow_sample=True,
                              readiness=report, resume=output / 'last.ckpt')
        check('학습 경로', f'{name} 재개 후 3 epoch', len(more) == 3 and more[-1]['epoch'] == 3, [h['epoch'] for h in more])
        value, _ = load_pair(index.role('validation')[0])
        source = value['source_canonical']
        translated = loaded.eval().translate(source)
        translated.validate()
        check('handoff', f'{name} 실제 CanonicalState 변환',
              isinstance(translated, CanonicalState) and translated.spatial_memory.shape == source.spatial_memory.shape
              and translated.spatial_memory.dtype == torch.bfloat16 and translated.object_pointer.dtype == torch.float32
              and torch.equal(translated.validity, source.validity)
              and torch.equal(translated.presence_logits, source.presence_logits),
              translated.metadata['translation'])
    print('sample validation state_loss', INFO['sample_validation'], flush=True)


def legacy(ctx):
    files = sorted(LEGACY_DIR.glob('*.pth'))
    check('기존 가중치', '선택된 Transformer 가중치 존재', len(files) == 1, files)
    raw = torch.load(files[0], map_location='cpu', weights_only=True)
    payload = raw.get('transformer', raw)
    model = restore_model('transformer', payload)
    fresh = build('transformer')
    a, b = model.state_dict(), fresh.state_dict()
    check('기존 가중치', 'strict 복원·key·shape가 transformer.json과 같음',
          list(a) == list(b) and all(a[k].shape == b[k].shape for k in a), len(a))
    check('기존 가중치', '구조 설정이 transformer.json과 같음', model.config == fresh.config,
          {k: (v, getattr(fresh.config, k)) for k, v in model.config.to_dict().items() if v != getattr(fresh.config, k)})
    check('기존 가중치', '파라미터 수', model.parameter_count() == 457216, model.parameter_count())
    INFO['legacy'] = dict(path=str(files[0]), preset=model.preset)
    return model


def full_validation(ctx, model):
    official = ctx['official']
    config = TrainConfig(workers=4, pin_memory=True, microbatch=64, global_batch=64)
    started = time.monotonic()
    model = model.to(device=DEVICE, dtype=torch.float32)
    metrics = evaluate(model, official.role('validation'), config, official.normalization['scales'], DEVICE)
    INFO['legacy']['full_validation'] = metrics
    INFO['legacy']['full_validation_seconds'] = round(time.monotonic() - started, 1)
    print('official validation', json.dumps(metrics)[:1200], flush=True)
    check('기존 가중치', '공식 validation 전체 평가 완료', metrics['all']['state_loss'] > 0, metrics['all'].get('state_loss'))


def main():
    global DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INFO['environment'] = dict(torch=torch.__version__, device=str(DEVICE),
                               gpu=torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else None,
                               visible=os.environ.get('CUDA_VISIBLE_DEVICES'))
    print(INFO['environment'], flush=True)
    guarded(structure)
    guarded(init_forward)
    guarded(roundtrip)
    guarded(weight_decay_groups)
    ctx = guarded(prepare_real)
    if ctx is not None:
        guarded(real_device, ctx)
        guarded(real_pipeline, ctx)
        model = guarded(legacy, ctx)
        if model is not None and '--skip-full-validation' not in sys.argv:
            guarded(full_validation, ctx, model)
    failed = [r for r in RESULTS if not r['ok']]
    (WORK / 'report.json').write_text(json.dumps(dict(results=RESULTS, info=INFO), ensure_ascii=False, indent=1, default=str))
    print(f'\nTOTAL {len(RESULTS)} PASS {len(RESULTS) - len(failed)} FAIL {len(failed)}', flush=True)
    for item in failed:
        print('FAILED:', item['group'], '|', item['name'], '|', item['detail'][:900], flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
