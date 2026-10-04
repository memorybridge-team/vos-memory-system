# Affine–Transformer 비교 설정 v1

Transformer는 `vos-memory-translator-nonlinear`의 `feat/lvos-ddp-training`, commit `2a57d8552f33f25adefe17ea51b1aebc40ecd1bc`의 `lvos_ddp.py`로 이미 학습됐다. 이 폴더는 같은 조건으로 **Affine만** 학습하고 평가한다. 확정 설정은 `configs/affine_comparable_v1.json`이다. 기존 `training.py`와 `budget_pipeline.py`의 pair 단위 학습은 이 비교 설정과 다르다.

| 항목 | 고정값 |
|---|---|
| 데이터 | Transformer가 쓴 LVOSv2 `raw_index.json`의 fit/development pair·유효 record 전체, 영상 분리 |
| 모델 | spatial 64→64 + pointer 256→256, bias 포함, 69,952 parameters |
| 초기화 | 두 가중치 I, bias 0 |
| Optimizer | AdamW, betas=(0.9, 0.999), eps=1e-8 |
| LR | 0.0003, 예정 update의 처음 5% 선형 warmup, cosine decay, floor 0.00001 |
| Weight decay | 가중치 0.0001, bias 0 |
| Update | 유효 record 64개, microbatch 16개 × 최대 4회 누적 |
| 마지막 batch | microbatch마다 loss × record 수를 더한 뒤 실제 유효 record 수로 나눔 |
| Loss | record별 spatial MSE / fit target spatial RMS² + record별 pointer MSE / fit target pointer RMS², record 평균 |
| Loss 가중치 | spatial 1, pointer 1, cosine 0 |
| 정규화 통계 | raw index에 기록된 fit-only RMS를 그대로 사용, 재계산 값과 상대오차 1e-7 이내 확인 |
| Gradient clipping | 전체 parameter gradient norm 1.0 |
| 학습 | 30 epoch, seed 7, FP32, augmentation 없음 |
| 순서 | epoch마다 seed+epoch으로 case 순서 shuffle; case 안 record 순서 유지; window는 case 경계를 넘음 |
| 검증·선택 | 매 epoch 전체 development record 평균 loss; 최소 loss epoch, 동점이면 나중 epoch |
| 저장 | 매 epoch 가중치를 `epochs/affine-epoch-NNNNN.pt`에 보존; `affine.pt`는 선택 epoch |
| 종료 | J&F early stopping 없음; 30 epoch 완료가 정식 비교 조건 |

## 기준 Transformer run

`runpod_comparison_evidence/20261004_corrected/`가 비교 기준이다(RunPod `CMMT-lvos-isolated/2a57d85-corrected-20261004T022603Z/`).

| 항목 | 값 |
|---|---|
| 상태 | `REAL_DDP_TRAIN_COMPLETED`, 2 GPU, microbatch 16 × accumulation 2 |
| raw index SHA-256 | `07e50f08a01cffb4495b5746d57fe8d034271f4854d612876b96168532895860` |
| 데이터 | fit 1,488 cases / 23,808 records, development 315 cases / 5,040 records, 제외 0, 영상 347/73 분리 |
| fit RMS | spatial 0.7997500379358756, pointer 0.5666758775201813 |
| Update | epoch당 372, 30 epoch 합계 11,160 |
| 선택 checkpoint | dev state loss 최소 epoch 30(loss 0.4269709), `translator/epoch-00030.pth` SHA-256 `8931de3f…bbcc` |
| J&F 평가 | 아직 없음. J&F early stopping·VOS best model 선정 없음 |

`--transformer-run`을 이 `training/` 폴더로 주면 위 조건을 학습 전에 검사한다. 이 폴더에는 `epoch-00030.pth` 가중치가 없으므로 `attach-transformer`에는 RunPod에서 그 파일을 추가로 받아야 한다. 참조 DDP는 2 GPU이므로 Affine 단일 GPU 학습과 수식은 같지만 rank 간 합산 순서 때문에 비트 단위로 같지는 않다.

RMS는 **loss의 상대 가중치**에만 사용한다. 전달할 memory 자체를 RMS로 나누지 않는다. 공간 pixel을 샘플링하지 않고 전체 memory frame을 사용한다. Presence 등 비학습 상태는 기존 handoff 계약을 따른다. GPU 수와 microbatch 분할은 비교 조건에 포함하지 않는다. Global batch, loss, 데이터, 학습 기회는 검사한다.

동점 규칙은 참조 DDP 학습기의 `best_state_loss.ckpt.json`과 같다. 이 파일은 dev loss가 그때까지의 최솟값과 같을 때마다 다시 쓰이므로 정확한 동점이면 나중 epoch가 남는다.

## 참조 학습기와의 일치 확인

참조 `lvos_ddp.py`는 모델을 Transformer `base`로 잠가서 Affine을 직접 학습할 수 없다. `tools/verify_reference_recipe.py`는 참조 커밋을 `git archive`로 꺼내, 모델 생성 함수와 `frozen_cases`만 바꿔 Affine을 학습시킨다. Raw cache 형식, `build_index`, case shuffle, window 구성, loss, gradient 누적·나눗셈, clipping, AdamW, scheduler, development 평가는 참조 코드 그대로다. 같은 `raw_index.json`으로 test10을 학습한 뒤 epoch별 update 수, record 수, dev loss, 가중치를 비교한다.

```bash
python test10/tools/verify_reference_recipe.py \
  --reference-repo /path/to/vos-memory-translator-nonlinear --epochs 30
```

합성 LVOS cache(fit 8 case 93 record, development 2 case 28 record, 30 epoch, CPU 1 thread)에서 모든 epoch의 dev loss와 가중치가 비트 단위로 같았다. `TEST10_REFERENCE_REPO`를 지정하면 같은 검사를 3 epoch로 unit test에서 반복한다.

Affine은 test10 translator 저장소의 `build_translator('linear')`로 만든다. 학습 시작 때 parameter 이름·shape, `output_dtype='source'`의 BF16 spatial handoff, 위치별 `Wx+b` 계산을 직접 검사한다. 저장소 revision은 `train_report.json`의 `model_contract`에 남긴다. 참조 커밋의 `LinearStateTranslator`는 `translate_tensors`가 없고 `build_translator('linear')`에서 실패하므로 Affine 정의로 쓰지 않는다.

## Affine 학습

기준 run이 있는 RunPod Volume에서는 다음 한 줄로 학습한다. `raw_index.json`의 원래 `/workspace` cache 경로를 그대로 쓰므로 cache를 복사하지 않는다. `TEST10_TRANSLATOR_REPO`는 test10 평가에 쓸 translator checkout이어야 한다. `2a57d85` checkout은 `linear` preset을 만들 수 없어 학습 시작 시 거절된다.

```bash
TEST10_TRANSLATOR_REPO=/workspace/test9/vos-memory-translator-nonlinear \
  bash test10/tools/train_affine_runpod.sh
```

기본값은 `REFERENCE_RUN=/workspace/CMMT-lvos-isolated/2a57d85-corrected-20261004T022603Z`, `OUTPUT_DIR=test10/training/affine_comparable_v1`이며 환경 변수로 바꿀 수 있다. 다른 머신에서는 아래처럼 원본 index와 cache 위치를 직접 지정한다.

Transformer 학습에 사용한 **원본 `raw_index.json`**과 Transformer run 폴더를 준비한다. 경로를 바꾸어야 하면 JSON을 편집하지 말고 `--pair-root`로 로컬 cache root를 지정한다. 이 root 아래 `fit/`, `development/`에 원래 이름의 `.pt`와 `.pt.sha256`을 둔다. Reader는 원본 v2 payload를 제한된 `weights_only=True` 로더로 읽고, checksum·Small/Base+ 모델 ID·tensor 대응·validity·video 분리를 검사한다.

```bash
python test10/comparable_training.py train \
  --index /data/reference/raw_index.json \
  --pair-root /data/lvos/cache \
  --transformer-run /data/reference/transformer_run \
  --output-dir test10/training/affine_comparable_v1 \
  --device cuda
```

`--transformer-run`은 `run.json`과 `metrics/history.json`만 읽고 Transformer 가중치는 읽지 않는다. 학습 전에 raw index SHA(또는 `--reference-index`로 같은 pair·fit/dev·유효 record), recipe, global batch, fit RMS, 30 epoch 완료, epoch별 fit/dev record 수와 update 수를 검사한다. 하나라도 다르면 학습을 시작하지 않는다. 검사 근거는 `train_report.json`의 `transformer_reference`에 남고, 평가 산출물에는 Transformer 파일을 요구하지 않는다. 생략하면 학습은 되지만 summary에 "Transformer run 대조: 없음"으로 표시된다.

기본 method는 `affine`이다. CPU cache 상한 기본값은 2 GiB다. `--budget-hours`로 시간 상한을 둘 수 있으나 30 epoch 전에 멈추면 `status=incomplete`이고 정식 비교에 쓸 수 없다. 중간 epoch resume는 없고 새 폴더에서 다시 시작한다.

원본 index가 없는 독립 실험은 `--index` 대신 `--pairs /data/lvos_pair_selection.json`을 쓸 수 있다. 형식은 `test10.pair_selection.v1`이며 LVOSv2만 허용한다. 이 경우 `--reference-index`로 Transformer index와 같은 데이터인지 검사한다. 정규화는 fit target에서 다시 계산하고, case 순서가 index와 다르면 shuffle 궤적도 달라진다. MOSE/LVOS 혼합 1,000/200 subset은 동일 데이터 조건이 아니다.

## 비교 checkpoint

`affine.pt`는 위의 dev state loss 규칙으로 고른 epoch다. 이미 나온 Transformer 결과가 다른 규칙으로 고른 checkpoint라면(예: 30 epoch 마지막, 또는 평가팀이 다른 기준으로 고른 epoch) Affine도 같은 규칙의 epoch를 쓴다.

```bash
python test10/comparable_training.py select \
  --affine-dir test10/training/affine_comparable_v1 \
  --epoch 30 --reason "Transformer 결과가 epoch 30 export" \
  --output-dir test10/training/affine_comparable_v1_epoch30
```

`select`는 저장된 epoch 가중치와 checksum만 복사하고 이유를 기록한다. test10 GT 점수로 epoch를 고르지 않는다.

## Affine 평가

평가 manifest는 `test10.v1` 형식이며 train/development와 겹치지 않는 영상으로 준비한다. 겹치면 audit에서 거절한다. 전환 후 최소 10개 처리 프레임이 필요하다.

```bash
python test10/run.py --stage audit \
  --selection /data/evaluation_selection.json \
  --training-dir test10/training/affine_comparable_v1 \
  --run-dir test10/runs/affine_comparable_v1 \
  --methods affine
python test10/run.py --stage smoke --run-dir test10/runs/affine_comparable_v1 --resume
python test10/run.py --stage full --run-dir test10/runs/affine_comparable_v1 --resume
```

`--methods affine`이면 Transformer·MLP checkpoint를 요구하지 않는다. Self-injection gate는 항상 실행된다. 같은 run에 Small-only, Base+-native, Direct 기준선이 필요하면 `--methods small_only base_native direct affine`으로 지정한다. 기존 `heldout.v1` manifest는 native 두 방법을 요구한다. Smoke는 포함된 데이터셋마다 최대 3개 case를 길이 구간별로 고정 선택한다. Full 평가 범위는 시간 예산과 고정 schedule을 따르므로 coverage를 결과와 함께 확인한다.

`summary.json`과 `summary.md`에 학습 설정, 비교 checkpoint epoch와 선택 규칙, Transformer run 대조 여부가 기록된다. `switch_frames.csv`에 +1..+10 J/F/J&F가 남는다.

**평가 조건은 두 모델이 같아야 비교가 성립한다.** 기준 Transformer run에는 아직 J&F 평가가 없다. Transformer도 같은 `evaluation_selection.json`, 같은 test10 하네스(test9 `mvp_scoring`, 객체별 독립 실행, +1..+10과 suffix J&F)로 평가한다. 한 run에서 같이 추론하려면 `attach-transformer`로 묶은 training directory와 `--methods affine transformer`를 쓴다. 참조 저장소 `lvos_evaluation.py`(development 3개 switch 비율, full replay 대비 retention)로 낸 Transformer 점수와 test10 Affine 점수는 직접 비교할 수 없다.

## 학습된 Transformer 연결(선택)

같은 test10 run에서 Transformer도 다시 추론하려는 경우에만 쓴다. Affine 단독 평가에는 필요 없다.

```bash
python test10/comparable_training.py attach-transformer \
  --affine-dir test10/training/affine_comparable_v1 \
  --transformer-run /data/reference/transformer_run \
  --output-dir test10/training/affine_transformer_comparable_v1
```

학습 전 검사와 같은 조건을 확인한 뒤, 같은 dev state loss 규칙의 Transformer export(`translator/epoch-NNNNN.pth`와 `.json`)를 strict load해 새 폴더에 담는다. `best_state_loss.ckpt.json`이 있으면 그 epoch와도 대조한다.
