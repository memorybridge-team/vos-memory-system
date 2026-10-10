"""SAM2 켜기 · 프롬프트 주기 · 추적 · 기억 꺼내기/넣기.

다른 파일은 아래 것만 쓴다 (SAM2 내부 구조는 몰라도 됨):

    runner  = load_runner("small")            # 모델 켜기
    session = runner.start(video)             # 영상 하나 열기 (세션 하나 = 객체 하나)
    session.add_prompt(frame, mask)           # "이 프레임에서 이 모양을 따라가라"
    session.load_memory(entries)              # 다른 세션에서 꺼낸 기억을 넣기
    session.encode_prompts()                  # 넣은 프롬프트를 기억 칸으로 바꿔 두기 (track 전 준비)
    for out in session.track(first, last):    # first ~ last 프레임 추적
        out.frame, out.mask, out.visible
    entries = session.export_memory()         # 지금 들고 있는 기억 꺼내기
    session.memory_of(frame)                  # 한 프레임의 기억 칸 (통계용)

SAM2 기억 = inference_state["output_dict_per_obj"][객체]:
    "cond_frame_outputs"      {프레임: 칸}   프롬프트를 받은 프레임
    "non_cond_frame_outputs"  {프레임: 칸}   스스로 추적한 프레임
    칸 = maskmem_features (1,64,64,64) · maskmem_pos_enc [(1,64,64,64)] · pred_masks (1,1,256,256)
         · obj_ptr (1,256) · object_score_logits (1,1)
새 프레임을 볼 때 SAM2가 읽는 것 = 프롬프트 프레임 칸 전부 + 최근 6장 maskmem + 최근 15장 obj_ptr.
그래서 최근 MEMORY_WINDOW 장보다 오래된 칸은 지운다 (긴 영상에서 GPU 메모리 절약, 결과는 같음).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from threading import Lock

import numpy as np
import torch
from PIL import Image

import settings

os.environ.setdefault("TQDM_DISABLE", "1")   # SAM2가 프레임마다 찍는 진행 막대 끄기

OBJ_ID = 1                                    # 세션 하나에 객체 하나만 둔다
MEMORY_FIELDS = ("maskmem_features", "maskmem_pos_enc", "pred_masks",
                 "obj_ptr", "object_score_logits")


@dataclass
class FrameOut:
    frame: int
    mask: np.ndarray    # bool (높이, 너비), 원본 해상도
    visible: bool | None  # 기본: presence > 0. benchmark에서 조회를 생략하면 None


def load_runner(model_key: str) -> "SAM2Runner":
    return SAM2Runner(model_key)


class SAM2Runner:
    def __init__(self, model_key: str):
        from sam2.build_sam import build_sam2_video_predictor

        cfg = settings.MODELS[model_key]
        checkpoint = str(Path(settings.SAM2_CHECKPOINT_DIR) / cfg["checkpoint"])
        self.name = model_key
        self.predictor = build_sam2_video_predictor(cfg["config"], checkpoint,
                                                    device=settings.DEVICE)
        from model.sam2_check import window_is_enough
        if not window_is_enough({"num_maskmem": self.predictor.num_maskmem,
                                "max_obj_ptrs_in_encoder": self.predictor.max_obj_ptrs_in_encoder,
                                "memory_temporal_stride_for_eval": self.predictor.memory_temporal_stride_for_eval}):
            raise ValueError("MEMORY_WINDOW가 설치된 SAM2의 기억 조회 범위보다 짧습니다.")

    def start(self, video) -> "Session":
        return Session(self, video)


class CPUFrameCache:
    """객체 실행 안에서만 공유하는 정규화된 CPU 입력 LRU. 반환 tensor는 독립 복사다."""

    def __init__(self, limit_mb):
        self.limit = int(limit_mb * 1024 ** 2)
        if self.limit < 0:
            raise ValueError('RGB_CACHE_MB는 0 이상이어야 합니다.')
        self.entries = OrderedDict()
        self.nbytes = 0
        self.lock = Lock()

    def get(self, key):
        with self.lock:
            value = self.entries.pop(key, None)
            if value is None:
                return None
            self.entries[key] = value
            return value.clone()

    def put(self, key, value):
        size = value.numel() * value.element_size()
        if not self.limit or size > self.limit:
            return
        with self.lock:
            old = self.entries.pop(key, None)
            if old is not None:
                self.nbytes -= old.numel() * old.element_size()
            while self.entries and self.nbytes + size > self.limit:
                _, old = self.entries.popitem(last=False)
                self.nbytes -= old.numel() * old.element_size()
            self.entries[key] = value.detach().clone()
            self.nbytes += size


class LazyFrames:
    """프레임을 필요할 때 한 장씩 읽는다.

    SAM2 기본 init_state 는 영상 전체를 한 번에 올려서 긴 영상(LVOS)에서 메모리가 터진다.
    읽는 방식(정사각형으로 줄이기, 평균·표준편차 정규화)은 SAM2 load_video_frames 와 같다.
    """

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, paths, image_size: int):
        self.paths = list(paths)
        self.image_size = image_size
        self._executor = None
        self._pending = None
        self._after = None
        self._last = -1
        self.cache = None
        self.profile = {'read_count': 0, 'read_prepare_seconds': 0.0,
                        'cache_hits': 0, 'cache_misses': 0, 'cache_bypasses': 0}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self._pending is not None:
            frame, future = self._pending
            self._pending = None
            if frame == i:
                x = future.result()
            else:
                # 이전 작업이 측정 프레임의 I/O와 겹치지 않도록 먼저 끝낸다.
                future.result()
                x = self._read(i)
        else:
            x = self._read(i)
        if self._executor is not None and self._after < i < self._last:
            self._pending = (i + 1, self._executor.submit(self._read, i + 1))
        return x

    def _read(self, i):
        # 이 worker는 CPU 이미지 준비만 한다. CUDA와 SAM2 상태에는 접근하지 않는다.
        use_cache = self.cache is not None and self._after is not None and i > self._after
        key = None
        if use_cache and self.cache.limit:
            path = Path(self.paths[i]).resolve()
            stat = path.stat()
            key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                   self.image_size, tuple(self.MEAN.flatten().tolist()), tuple(self.STD.flatten().tolist()))
            value = self.cache.get(key)
            if value is not None:
                self.profile['cache_hits'] += 1
                return value
            self.profile['cache_misses'] += 1
        else:
            self.profile['cache_bypasses'] += 1
        t0 = perf_counter()
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB").resize((self.image_size, self.image_size))
            x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = (x - self.MEAN) / self.STD
        self.profile['read_count'] += 1
        self.profile['read_prepare_seconds'] += perf_counter() - t0
        if key is not None:
            self.cache.put(key, x)
        return x

    def configure_prefetch(self, measure_through, last):
        """측정 마지막 프레임 뒤에서만 다음 한 장을 준비한다."""
        self.close()
        self._after, self._last = measure_through, last
        if settings.PREFETCH_FRAMES and last > measure_through:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='vos-frame')

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor, self._pending = None, None


def _copy_to(x, device):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [_copy_to(v, device) for v in x]
    # copy=True는 같은 장치에서도 독립된 저장 공간을 보장한다.
    # 다른 장치로 옮긴 뒤 clone하는 두 번째 전체 복사를 피한다.
    return x.detach().to(device=device, copy=True)


class Session:
    def __init__(self, runner: SAM2Runner, video):
        import sam2.sam2_video_predictor as svp

        self.predictor = runner.predictor
        self.skip_visible = False
        frames = LazyFrames(video.frame_paths, self.predictor.image_size)
        self.frames = frames
        height, width = video.size
        # init_state 가 프레임을 전부 읽지 않도록, 읽는 함수만 잠깐 바꿔 끼운다.
        original = svp.load_video_frames
        svp.load_video_frames = lambda **_: (frames, height, width)
        try:
            with self._context():
                self.state = self.predictor.init_state(
                    video_path=str(video.frame_paths[0].parent),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=settings.OFFLOAD_STATE_TO_CPU)
        finally:
            svp.load_video_frames = original
        with self._context():
            self.obj_idx = self.predictor._obj_id_to_idx(self.state, OBJ_ID)

    # ── 기본 동작 ──────────────────────────────────────────────────────
    @contextmanager
    def _context(self):
        with torch.inference_mode():
            if settings.USE_BF16 and torch.cuda.is_available():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    yield
            else:
                yield

    def _store(self) -> dict:
        return self.state["output_dict_per_obj"][self.obj_idx]

    def add_prompt(self, frame: int, mask: np.ndarray) -> None:
        with self._context():
            self.predictor.add_new_mask(self.state, frame_idx=frame, obj_id=OBJ_ID,
                                        mask=torch.as_tensor(mask, dtype=torch.bool))

    def encode_prompts(self) -> None:
        """넣은 프롬프트 마스크를 기억 칸으로 바꿔 둔다 (SAM2 propagate_in_video_preflight).

        SAM2 는 이 일을 track 의 첫 프레임 때 한다. 전환 지연(준비 시간)에 넣으려고 미리 부른다.
        track 이 다시 불러도 할 일이 없어서 결과는 같다.
        """
        with self._context():
            self.predictor.propagate_in_video_preflight(self.state)

    def track(self, first: int, last: int):
        """first ~ last 프레임을 차례로 추적하며 한 장씩 돌려준다."""
        frames = self.predictor.propagate_in_video(
            self.state, start_frame_idx=first, max_frame_num_to_track=last - first)
        while True:
            with self._context():
                try:
                    frame, _, logits = next(frames)
                except StopIteration:
                    return
                mask = (logits[0, 0] > 0).cpu().numpy()
                visible = None if self.skip_visible else self._visible(frame)
                self._forget_old(frame)
            yield FrameOut(frame, mask, visible)

    def configure_prefetch(self, measure_through: int, last: int):
        self.frames.configure_prefetch(measure_through, last)

    def configure_benchmark(self, frame_cache=None):
        """일반 Session 기본 동작은 유지하고 benchmark에서만 선택한다."""
        self.frames.cache = frame_cache
        self.skip_visible = settings.BENCHMARK_SKIP_VISIBLE

    def close(self) -> None:
        self.frames.close()
        self.state = None

    # ── 기억 ──────────────────────────────────────────────────────────
    def memory_of(self, frame: int) -> dict | None:
        store = self._store()
        return store["cond_frame_outputs"].get(frame) or store["non_cond_frame_outputs"].get(frame)

    def _visible(self, frame: int) -> bool:
        out = self.memory_of(frame)
        return out is not None and float(out["object_score_logits"].max()) > 0

    def _forget_old(self, frame: int) -> None:
        non_cond = self._store()["non_cond_frame_outputs"]
        for t in [t for t in non_cond if t < frame - settings.MEMORY_WINDOW]:
            del non_cond[t]

    def export_memory(self) -> dict:
        """지금 들고 있는 기억 칸 전부를 CPU로 복사해 꺼낸다: {프레임: {칸 필드..., "is_cond"}}"""
        entries = {}
        with self._context():
            for key, is_cond in (("cond_frame_outputs", True), ("non_cond_frame_outputs", False)):
                for frame, out in self._store()[key].items():
                    entry = {name: _copy_to(out[name], "cpu") for name in MEMORY_FIELDS}
                    entry["is_cond"] = is_cond
                    entries[frame] = entry
        return entries

    def export_features(self) -> dict:
        """복원율용 두 기억 필드만 CPU로 복사한다. 전환 비용 측정 밖에서 호출한다."""
        entries = {}
        with self._context():
            for key, is_cond in (("cond_frame_outputs", True), ("non_cond_frame_outputs", False)):
                for frame, out in self._store()[key].items():
                    entries[frame] = {name: _copy_to(out.get(name), "cpu")
                                      for name in ("maskmem_features", "obj_ptr")}
                    entries[frame]["is_cond"] = is_cond
        return entries

    def load_memory(self, entries: dict) -> None:
        """꺼낸 기억 칸을 이 세션에 넣는다. 넣은 뒤 track(전환 프레임 + 1, ...) 으로 이어간다.

        SAM2가 칸을 읽는 위치에 맞춰 둔다: maskmem_features·pred_masks 는 저장 장치,
        나머지는 계산 장치 (SAM2 _run_single_frame_inference 와 같은 배치).
        """
        store = self._store()
        tracked = self.state["frames_tracked_per_obj"][self.obj_idx]
        compute, storage = self.state["device"], self.state["storage_device"]
        with self._context():
            for frame, entry in sorted(entries.items()):
                out = {
                    "maskmem_features": _copy_to(entry["maskmem_features"], storage),
                    "maskmem_pos_enc": _copy_to(entry["maskmem_pos_enc"], compute),
                    "pred_masks": _copy_to(entry["pred_masks"], storage),
                    "obj_ptr": _copy_to(entry["obj_ptr"], compute),
                    "object_score_logits": _copy_to(entry["object_score_logits"], compute),
                }
                key = "cond_frame_outputs" if entry["is_cond"] else "non_cond_frame_outputs"
                store[key][frame] = out
                tracked[frame] = {"reverse": False}
