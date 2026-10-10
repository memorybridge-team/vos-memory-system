"""설치된 SAM2 구조 확인.

본 실험 전에 확인할 세 가지:
  1. Small / Base+ 기억 칸의 모양이 같은가 → 같아야 Direct State Copy 가 된다.
  2. 기억 위치 정보(maskmem_pos_enc)가 두 모델에서 같은 값인가 → 같으면 복사해도 문제 없음.
  3. 같은 모델에서 기억을 꺼냈다 새 세션에 넣고 이어가도, 끊지 않은 결과와 프레임마다 같은가
     → 같아야 "꺼내기/넣기" 코드가 기억을 빠짐없이 옮긴다는 뜻.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

import settings

REQUIRED_STATE_KEYS = ("output_dict_per_obj", "temp_output_dict_per_obj",
                       "frames_tracked_per_obj", "obj_id_to_idx", "device", "storage_device")


def _shape(x):
    if isinstance(x, (list, tuple)):
        return [_shape(v) for v in x]
    return f"{tuple(x.shape)} {str(x.dtype).replace('torch.', '')}"


def describe(runner, video, obj_id: int, start: int, n_frames: int = 8) -> dict:
    """모델 설정값 + 기억 칸 모양. 몇 프레임만 추적해 본다."""
    p = runner.predictor
    session = runner.start(video)
    missing = [k for k in REQUIRED_STATE_KEYS if k not in session.state]
    session.add_prompt(start, video.object_mask(start, obj_id))
    last = min(start + n_frames, video.num_frames - 1)
    for _ in session.track(start, last):
        pass
    entry = session.memory_of(last)
    info = {
        "model": runner.name,
        "image_size": p.image_size,
        "num_maskmem": p.num_maskmem,
        "max_obj_ptrs_in_encoder": p.max_obj_ptrs_in_encoder,
        "memory_temporal_stride_for_eval": p.memory_temporal_stride_for_eval,
        "hidden_dim": p.hidden_dim,
        "mem_dim": p.mem_dim,
        "missing_state_keys": missing,
        "fields": {name: _shape(entry[name]) for name in entry},
        "entry": entry,
    }
    session.close()
    return info


def shape_mismatches(a: dict, b: dict) -> list[str]:
    """두 모델의 기억 칸 모양이 다른 곳."""
    return [f"{name}: {a['fields'][name]} vs {b['fields'].get(name)}"
            for name in a["fields"] if a["fields"][name] != b["fields"].get(name)]


def same_pos_enc(a: dict, b: dict) -> bool:
    pa = [t.float().cpu() for t in a["entry"]["maskmem_pos_enc"]]
    pb = [t.float().cpu() for t in b["entry"]["maskmem_pos_enc"]]
    return len(pa) == len(pb) and all(torch.equal(x, y) for x, y in zip(pa, pb))


def window_is_enough(info: dict) -> bool:
    """MEMORY_WINDOW 가 SAM2가 실제로 읽는 범위를 덮는가."""
    stride = info.get("memory_temporal_stride_for_eval", 1)
    spatial_need = max(0, (info["num_maskmem"] - 2) * stride + 1)
    need = max(spatial_need, info["max_obj_ptrs_in_encoder"] - 1)
    return settings.MEMORY_WINDOW >= need


def roundtrip(runner, video, obj_id: int, start: int, cut: int, last: int) -> dict:
    """끊지 않고 쭉 간 결과 vs cut 에서 기억을 꺼내 새 세션에 넣고 이어간 결과."""
    if not 0 <= start <= cut < last < video.num_frames:
        raise ValueError("roundtrip에는 유효한 전반 구간과 1프레임 이상의 이어 추적 구간이 필요합니다.")
    prompt = video.object_mask(start, obj_id)

    whole = runner.start(video)
    try:
        whole.add_prompt(start, prompt)
        reference = {out.frame: out.mask for out in whole.track(start, last)}
    finally:
        whole.close()

    first = runner.start(video)
    try:
        first.add_prompt(start, prompt)
        for _ in first.track(start, cut):
            pass
        memory = first.export_memory()
    finally:
        first.close()

    second = runner.start(video)
    try:
        second.load_memory(memory)
        resumed = {out.frame: out.mask for out in second.track(cut + 1, last)}
    finally:
        second.close()

    frames = sorted(resumed)
    expected = set(range(cut + 1, last + 1))
    comparable = sorted(expected & set(resumed) & set(reference))
    same = [f for f in comparable if np.array_equal(resumed[f], reference[f])]
    diff_pixels = max((int((resumed[f] != reference[f]).sum()) for f in comparable), default=0)
    missing = sorted(expected - set(resumed))
    missing_reference = sorted(expected - set(reference))
    unexpected = sorted(set(resumed) - expected)
    return {"frames": len(frames), "expected_frames": len(expected),
            "identical_frames": len(same), "max_diff_pixels": diff_pixels,
            "missing_frames": missing, "missing_reference_frames": missing_reference,
            "unexpected_frames": unexpected,
            "passed": not (missing or missing_reference or unexpected) and len(same) == len(expected)}


def report_path() -> Path:
    return Path(settings.OUTPUT_ROOT) / 'checks' / 'sam2.json'


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_signature() -> dict:
    """장치·모델·설치 코드가 바뀌면 이전 검사 통과 기록을 사용할 수 없다."""
    sam2 = importlib.import_module('sam2')
    package = Path(sam2.__file__).resolve().parent
    files = sorted(p for p in package.rglob('*')
                   if p.is_file() and p.suffix in ('.py', '.yaml', '.so'))
    code = hashlib.sha256()
    for path in files:
        code.update(str(path.relative_to(package)).encode())
        code.update(_digest(path).encode())
    root = Path(__file__).resolve().parents[1]
    device = torch.device(settings.DEVICE)
    gpu = None
    if device.type == 'cuda':
        gpu = {'name': torch.cuda.get_device_name(device),
               'capability': list(torch.cuda.get_device_capability(device))}
    return {
        'device': str(device), 'gpu': gpu, 'torch': str(torch.__version__),
        'cuda': torch.version.cuda, 'use_bf16': settings.USE_BF16,
        'offload_state_to_cpu': settings.OFFLOAD_STATE_TO_CPU,
        'memory_window': settings.MEMORY_WINDOW,
        'source_model': settings.SOURCE_MODEL, 'target_model': settings.TARGET_MODEL,
        'models': {key: {**settings.MODELS[key], 'checkpoint_sha256': _digest(
            Path(settings.SAM2_CHECKPOINT_DIR) / settings.MODELS[key]['checkpoint'])}
            for key in (settings.SOURCE_MODEL, settings.TARGET_MODEL)},
        'sam2_path': str(package), 'sam2_sha256': code.hexdigest(),
        'benchmark_code': {name: _digest(root / name) for name in
                           ('model/sam2_runner.py', 'model/sam2_check.py', 'scripts/0_check_sam2.py')},
    }


def check_names() -> set[str]:
    return {'shapes', 'pos_enc'} | {
        f'{key}/{test}' for key in (settings.SOURCE_MODEL, settings.TARGET_MODEL)
        for test in ('state_keys', 'window', 'dtypes', 'roundtrip')}


def save_passed(checks: dict, roundtrips: dict, case: dict) -> Path:
    if set(checks) != check_names() or not all(value is True for value in checks.values()):
        raise ValueError("SAM2 검사에 실패했습니다. 본평가를 시작할 수 없습니다.")
    payload = {'schema': 1, 'passed': True, 'checks': checks, 'roundtrips': roundtrips,
               'case': case, 'runtime': runtime_signature(),
               'checked_at': datetime.now(timezone.utc).isoformat()}
    path = report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{uuid4().hex}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def require_passed() -> None:
    path = report_path()
    message = "python scripts/0_check_sam2.py를 현재 환경에서 먼저 실행하세요."
    if not path.exists():
        raise ValueError(f"SAM2 검사 통과 기록이 없습니다. {message}")
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise ValueError(f"SAM2 검사 기록을 읽을 수 없습니다. {message}") from exc
    if (not isinstance(payload, dict) or payload.get('schema') != 1
            or payload.get('passed') is not True
            or payload.get('checks') != dict.fromkeys(check_names(), True)):
        raise ValueError(f"SAM2 검사를 모두 통과하지 않았습니다. {message}")
    if payload.get('runtime') != runtime_signature():
        raise ValueError(f"SAM2 검사 이후 모델·코드·장치 설정이 바뀌었습니다. {message}")
