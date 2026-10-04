# Affine–Transformer 비교 설정 v1

`comparable_training.py`는 LVOS에서 Affine만 새로 학습하는 기본 진입점이다. 확정 설정은 `configs/affine_comparable_v1.json`이다. 기존 `training.py`와 `budget_pipeline.py`의 pair 단위 학습은 이 비교 설정과 다르다.

| 항목 | 고정값 |
|---|---|
| 데이터 | LVOSv2의 동일 fit/development pair·유효 record 전체, 영상 분리 |
| 모델 | spatial 64→64 + pointer 256→256, bias 포함, 69,952 parameters |
| 초기화 | 두 가중치 I, bias 0 |
| Optimizer | AdamW, betas=(0.9, 0.999), eps=1e-8 |
| LR | 0.0003, 예정 update의 처음 5% 선형 warmup, cosine decay, floor 0.00001 |
| Weight decay | 가중치 0.0001, bias 0 |
| Update | 유효 record 64개, microbatch 16개 × 최대 4회 누적 |
| 마지막 batch | 실제 record 개수로 gradient 평균, 버리거나 복제하지 않음 |
| Loss | spatial MSE / fit target spatial RMS² + pointer MSE / fit target pointer RMS² |
| Loss 가중치 | spatial 1, pointer 1, cosine 0 |
| 정규화 통계 | fit target 전체에서 FP64로 계산, development 통계 사용 안 함 |
| Gradient clipping | 전체 parameter gradient norm 1.0 |
| 학습 | 30 epoch, seed 7, FP32, augmentation 없음 |
| 순서 | epoch마다 seed+epoch으로 pair 순서 shuffle; pair 안의 유효 record 대응 보존 |
| 검증·선택 | 매 epoch 전체 development record 평균 loss, 최소 loss checkpoint; 동점은 이른 epoch |
| 종료 | J&F early stopping 없음; 30 epoch 완료가 정식 비교 조건 |
| 평가 | 동일 case/Small state/Base+ weights, +1..+10 J/F/J&F와 전체 suffix |

RMS는 **loss의 상대 가중치**에만 사용한다. 전달할 memory 자체를 RMS로 나누지 않는다. 공간 pixel을 샘플링하지 않고 전체 memory frame을 사용한다. Presence 등 비학습 상태는 기존 handoff 계약을 따른다. Transformer는 구조의 전체 config와 weight를 export에서 strict load하며 현재 로컬 `base` 기본값으로 다시 해석하지 않는다.

참조 recipe는 `vos-memory-translator-nonlinear`의 `feat/lvos-ddp-training`, commit `2a57d8552f33f25adefe17ea51b1aebc40ecd1bc`에 정의된 state 학습 조건이다. GPU 수와 microbatch 분할 방식은 비교 일치 조건에 포함하지 않는다. Global batch, loss, 데이터와 학습 기회는 검사한다. 실제 RunPod 실행 파일은 아직 확인하지 않았으므로 기존 Transformer가 이 조건을 만족한다고 미리 판정하지 않는다.

## Affine만 학습

RunPod에서 실제 Transformer 학습에 사용한 **원본 `raw_index.json`**을 준비한다. 경로를 바꾸어야 하면 JSON을 편집하지 말고 `--pair-root`로 로컬 cache root를 지정한다. 이 root 아래 `fit/`, `development/`에 원래 이름의 `.pt`와 `.pt.sha256`을 둔다. 원본 index의 SHA를 보존하면 기존 Transformer와 데이터 일치를 바로 확인할 수 있다. 서로 다른 index라도 원본 `--reference-index`를 제공하면 파일 경로와 순서를 제외한 pair SHA·fit/dev 소속·유효 record가 같은지 비교한다. Reader는 원본 v2 payload를 제한된 `weights_only=True` 로더로 읽고, checksum·Small/Base+ 모델 ID·tensor 대응·validity·video 분리를 검사한다. 존재하지 않는 생성 이력을 만들어 넣지 않는다.

```bash
python test10/comparable_training.py train \
  --index /data/reference/raw_index.json \
  --pair-root /data/lvos/cache \
  --output-dir test10/training/affine_comparable_v1 \
  --device cuda
```

기본 method는 `affine`이며 MLP/Transformer를 함께 학습하지 않는다. CPU cache 상한 기본값은 2 GiB다. GPU microbatch 외의 CPU 텐서, 데이터 로딩과 Python 객체 메모리는 별도다.

원본 index가 없는 독립 실험은 `--index` 대신 `--pairs /data/lvos_pair_selection.json`을 사용할 수 있다. 형식은 기존 `test10.pair_selection.v1`이며 LVOSv2만 허용한다. 이 경우 기존 RunPod Transformer 연결 명령에 `--reference-index /data/reference/raw_index.json`을 추가해 같은 데이터인지 검사한다. MOSE/LVOS 혼합 1,000/200 subset은 이 LVOS 비교의 동일 데이터 조건이 아니다.

`--budget-hours 3.5` 같은 시간 상한을 추가할 수 있다. 30 epoch 전에 멈추면 `status=incomplete`이며 저장된 best checkpoint도 정식 완료 비교용으로 로드하지 않는다. 따라서 기존 5시간 목표와 30 epoch 완료를 동시에 보장하지 않는다. 실제 학습 시간은 실행 후 확인한다. 이 경로는 중간 epoch resume를 제공하지 않으며 중단 후 새 폴더에서 재시작한다.

## 학습된 Transformer 연결

RunPod run에서 다음 파일만 로컬에 준비하면 된다. 전체 run/모든 checkpoint 다운로드는 필요하지 않다.

- `run.json`
- `metrics/history.json`
- development loss가 최소인 epoch의 `translator/epoch-NNNNN.pth` 및 같은 이름 `.json`
- 학습에 사용했던 원본 `raw_index.json`은 위 Affine 학습 입력으로 사용

```bash
python test10/comparable_training.py attach-transformer \
  --affine-dir test10/training/affine_comparable_v1 \
  --transformer-run /data/reference/transformer_run \
  --output-dir test10/training/affine_transformer_comparable_v1
```

raw index SHA 또는 동일 pair·fit/dev·유효 record 목록, fit RMS, LR/weight decay/스케줄/loss/seed/FP32/global batch, 30개 완료 epoch, 각 epoch 전체 fit/dev record 수와 update 수를 검사한다. 그 후 dev loss가 최소인 Transformer export와 Affine checkpoint를 새 비교 폴더에 담는다. 설정·데이터·완료 학습량이 다르면 구체적인 차이를 출력하고 연결을 거절한다. 체크포인트의 선택 epoch는 서로 달라도 된다. 이것은 선언된 실행 기록과 artifact의 일치 검사이며, state loss가 낮다는 이유로 VOS 성능까지 검증된 것으로 취급하지 않는다.

## test10 평가

평가 manifest는 기존 `test10.v1` 형식이며 train/development와 겹치지 않는 영상으로 준비한다. 기본 seed는 7이다. 제공한 영상의 미래 프레임 범위를 고정하고 전환 후 최소 10개 처리 프레임을 포함한다. Affine 단독 결과를 먼저 얻으려면 아래처럼 실행한다.

```bash
python test10/run.py --stage audit \
  --selection /data/evaluation_selection.json \
  --training-dir test10/training/affine_comparable_v1 \
  --run-dir test10/runs/affine_comparable_v1 \
  --methods small_only base_native direct affine
python test10/run.py --stage smoke --run-dir test10/runs/affine_comparable_v1 --resume
python test10/run.py --stage full --run-dir test10/runs/affine_comparable_v1 --resume
```

Affine–Transformer 비교는 새 run directory와 연결한 training directory를 사용하고 audit의 methods에 `transformer`를 추가한다. MLP checkpoint는 요구하지 않는다. 순수 Affine 추론만 선택하려면 일반 manifest에서 `--methods affine`을 사용할 수 있지만 기존 `heldout.v1` manifest는 native 두 방법을 요구한다. 비교의 smoke는 포함된 데이터셋마다 최대 3개 case를 길이 구간별로 고정 선택한다. Smoke 이후 full 평가 범위는 기존 시간 예산과 고정 schedule을 따르므로 전체 후보 완료 여부를 결과의 coverage로 확인한다.

`summary.json`에 고정 학습 설정과 comparison protocol을 포함한다. `summary.md`, `switch_frames.csv`에 첫 10프레임 및 suffix 성능이 기록된다. `train_report.json`에는 실제 epoch/update/record 수, fit RMS, 선택 checkpoint, 외부 Transformer의 확인 근거가 남는다. test10 GT 점수로 checkpoint를 다시 고르지 않는다.
