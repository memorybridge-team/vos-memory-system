# Paired-state 수집 코드 묶음 — 2026-10-09

현재 로컬 Task 07 collector와 수정된 reference runtime을 함께 묶은 실행용 소스 snapshot이다. GitHub main과 동일하다고 가정하지 않는다. 원본 데이터셋·checkpoint·가상환경·cache는 별도로 준비한다.

## 실행 순서

```text
manifests/*.json (train + fit/development 또는 validation/external)
  → scripts/task07/prepare_paired_state_dataset.py
      → manifest_adapters.py (validation/external 입력 정규화)
      → cmmt-sam2-prepare-case
          → src/vos_memory_inspector/cli.py
          → roundtrip.py
              → runner.py (객체별 최초 GT prompt mask)
              → sam2_lazy_loader.py (RGB lazy loading)
              → SAM2 upstream Small/Base+ 각각 prompt부터 switch까지 추론
              → sam2_state.py + state_schema.py (export·active-memory·계약)
              → case_cache.py (CPU state-only cache + SHA-256)
```

양쪽 모델은 전체 영상 runtime 좌표를 유지한다. 원본 prompt/switch ID를 JPEG 순번으로 바꾸며 late prompt를 frame 0에 강제로 넣지 않는다. conditioning + 최근 15 non-conditioning 기록을 보존하고 미래 prediction mask는 저장하지 않는다. `paired_state_cache.py` compact 변환은 이 기본 수집 경로에서 호출하지 않는다.

## 설치와 단일 worker 예시

Python 3.10 이상, 해당 GPU/driver에 맞는 PyTorch와 SAM2 의존성이 필요하다. SAM2 upstream은 `2b90b9f5ceec907a1c18123530e92e794ad901a4`로 고정한다. 공식 SAM2 설치 안내에 따라 CUDA extension까지 설치한다. 실제 복구 실행 환경은 Python 3.12 / torch 2.8.0+cu128 / L4였으며 다른 환경의 bitwise 동일성을 보장하지 않는다.

ZIP을 풀고 묶음 최상위 디렉터리에서 실행한다.

```bash
python -m pip install -e .
python scripts/task07/prepare_paired_state_dataset.py \
  --source-manifest manifests/mosev2_train_v1.json \
  --fit-split manifests/mosev2_train_v1_fit.json \
  --development-split manifests/mosev2_train_v1_development.json \
  --dataset-root /workspace/CMMT/data/MOSEv2/extracted/train \
  --sam2-repo /workspace/CMMT/.external/sam2 \
  --source-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --source-checkpoint /workspace/CMMT/checkpoints/sam2.1_hiera_small.pt \
  --source-model-id sam2.1-small \
  --target-config configs/sam2.1/sam2.1_hiera_b+.yaml \
  --target-checkpoint /workspace/CMMT/checkpoints/sam2.1_hiera_base_plus.pt \
  --target-model-id sam2.1-base-plus \
  --network-volume-root /workspace/paired-output \
  --run-directory /workspace/paired-runs/worker-0
```

Validation/external은 위 train manifest 인자 3개 대신 `--selection-manifest manifests/lvosv2_valid_v1.json` 등을 사용하고 dataset root를 해당 split으로 지정한다. Root 아래 `JPEGImages/`와 `Annotations/`가 있어야 한다. External paired states는 평가 참조용이며 translator 학습·통계·모델 선택에 사용하지 않는다.

여러 GPU/Pod은 같은 manifest·checkpoint·소스·seed를 사용하고 `--dynamic-queue --queue-root /workspace/paired-queue --worker-id worker-0`을 추가한다. 각 worker의 run-directory와 worker-id는 서로 다르게 지정하고 queue/output만 공유한다. 한 Pod의 여러 GPU는 `CUDA_VISIBLE_DEVICES=0`, `1`, `2`로 각 worker를 실행한다.

## 묶음 내용과 검증 범위

- `src/`: reference runtime 전체 Python 모듈. CLI의 간접 import 때문에 함께 포함했다. legacy DAVIS 모듈이 있어도 이번 dataset manifest나 수집 입력에 DAVIS를 포함하는 것은 아니다.
- `scripts/task07/`: 현재 collector/adapter 및 별도 prompt 복구·CUDA preflight 도구. repair launcher는 당시 `/workspace` 경로를 사용하므로 현재 환경에 맞춰 확인한다.
- `manifests/`: MOSEv2/LVOS v2 train·fit·development·validation, VOST validation, PUMaVOS, M³-VOS frozen JSON. DAVIS manifest는 제외했다.
- `docs/`: prompt 정렬 복구 실행 안내와 완료 보고서.
- `FILE_INVENTORY.json`, `SHA256SUMS`: 각 파일의 원래 로컬 위치와 SHA-256. ZIP 생성 시 각 entry를 다시 읽어 원본 hash와 대조한다.

기존 prompt/cache 관련 CPU tests 22개와 collector tests 4개 통과, 복구된 실제 cache 38개 production 재검증 통과 기록을 포함한다. 전체 reference suite의 기존 M³-VOS inventory test 1개 실패는 남아 있으며 이번 묶음 전체의 새 GPU 수집 실험을 수행한 것은 아니다. 공식 SAM2 코드와 config YAML은 별도 upstream checkout에서 읽는다.
