# test10 — Small → Base+ Affine 상태 전환 실험

Small이 frame `t`까지 처리한 상태를 Base+에 전달해 `t+1`부터 이어 처리하는 방법을 비교한다. 결과 패키지에는 평가 harness, 선택 manifest, case별 점수 JSON, 요약 통계, 실행·비용·provenance 기록이 포함되어 있다.

| 구분 | 항목 | 설정 |
|---|---|---|
| 모델 | 변환 방식 | Spatial memory와 object pointer에 각각 `Wx+b` 적용 |
| 모델 | Spatial 변환 | 위치마다 같은 64 → 64 선형층 사용 |
| 모델 | Pointer 변환 | 256 → 256 선형층 |
| 모델 | 파라미터 수 | 69,952개 |
| 데이터 | 학습 데이터 | LVOSv2 fit: 1,488 pair / 347영상 |
| 데이터 | 검증 데이터 | LVOSv2 development: 315 pair / 73영상 |
| 데이터 | 유효 memory record | 학습 23,808개 / 검증 5,040개 |
| 데이터 | 학습 목표 | 같은 영상·객체·시점의 Small 상태 → Base+ 상태 |
| 데이터 | 증강 | 없음 |
| 손실 | 구성 | 정규화된 spatial MSE + 정규화된 pointer MSE |
| 손실 | 정규화 | 각 MSE를 학습 데이터의 target RMS²로 나눔 |
| 손실 | Target RMS | Spatial 0.7997500379 / Pointer 0.5666758775 |
| 손실 | 적용 범위 | 유효 record만 계산하며 padding 제외 |
| 손실 | 입력 정규화 | 하지 않음 — RMS는 손실 계산에만 사용 |
| 최적화 | Optimizer | AdamW |
| 최적화 | Learning rate | 0.0003 |
| 최적화 | AdamW 설정 | β₁ 0.9 β₂ 0.999 ε 1e−8 |
| 최적화 | Weight decay | 가중치 .0001 / bias 0 |
| 스케줄 | Learning rate 감소 | Cosine decay 최소 0.00001 |
| 실행 | Epoch | 30 조기 종료 없음 |
| 실행 | Update 수 | Epoch당 372회 총 11,160회 |
| 실행 | 학습 정밀도 | FP32 |
| 실행 | Seed | 7 |
| 모델 선택 | 검증 주기 | 매 epoch, 검증 유효 record 전체 평가 |
| 모델 선택 | 선택 기준 | 검증 normalized loss 최소 동점이면 나중 epoch |
| 모델 선택 | 선택 결과 | Epoch 28 / optimizer step 10,416 |
| 모델 선택 | 선택 시 검증 손실 | 0.452371 |


## 결과 요약

### 결과를 계산하는게 잘못되어 있다
### -> 최종 J&F가 아주 높은 이유가 전환 프레임 ~ 영상 종료시점 까지 진행한 다음의 결과이기 때문에 translator의 성능이라고 보기보다 base+의 성능이라고 봐야한다

- 4시간 누적 예산 중 **157.5분** 사용. Smoke 32 cases와 full schedule 116 cases, 총 **148 cases** 완료.
- MOSEv2 77 cases / 77 videos, LVOSv2 41 / 40, DAVIS 2017 30 / 30. 전체 147개 고유 영상이다. 객체별 독립 실행이며 다객체 동시 추적 결과로 해석하지 않는다.
- 13개 품질 방법 모두와 self-injection gate가 148 cases에서 완료됐고, 실패 attempt는 없다. Self-injection은 148/148 통과(logit max error 0, binary IoU 1.0).
- 기존 fit shard와 translator checkpoint는 재사용했고 새 학습은 하지 않았다. 과거 test9 점수 row는 provenance가 불충분해 재사용 0건으로 기록했다.

J&F (point, 데이터셋별 영상 평균):

| Dataset | Direct Copy | Affine `Wx+b` | Residual MLP | Transformer | Last-mask | Anchor Replay-16 |
|---|---:|---:|---:|---:|---:|---:|
| MOSEv2 | 31.42 | 64.85 | 60.12 | 64.90 | 60.69 | 65.98 |
| LVOSv2 | 3.64 | 94.26 | 95.24 | 95.21 | 95.01 | 86.72 |
| DAVIS 2017* | 10.42 | 90.02 | 90.08 | 89.97 | 90.24 | 84.61 |

Affine은 Direct Copy보다 크게 높았다. Nonlinear의 affine 대비 paired 차이는 MOSEv2에서 MLP −4.73 [−9.36, −1.05], Transformer +0.05 [−2.93, +2.71]; LVOSv2에서 MLP +0.99 [−0.07, +3.04], Transformer +0.96 [−0.12, +3.03]; DAVIS에서 각각 +0.06 [−0.97, +0.99], −0.05 [−0.20, +0.11] point였다. 따라서 nonlinear가 데이터셋 전반에서 affine보다 낫다는 근거는 확인되지 않았다.

Component ablation에서는 spatial-only affine이 full affine에 가까운 점수를 보였고, pointer-only affine은 Direct Copy와 비슷했다. 이번 조건에서는 성능 개선이 주로 spatial memory 변환에서 나왔다. Affine과 last-mask/replay 간 paired CI는 대체로 0을 포함해 일관된 우위가 확인되지 않았다.

동일한 32-case smoke 비용 비교에서 중앙 handoff 시간은 affine 약 7.4ms, Anchor Replay-16 약 560ms였다. 첫 suffix 출력까지는 각각 약 57ms, 604ms였다. 이는 warm-model 고정 smoke 계측이며 모델 load 및 video 초기화 시간은 별도 기록이다.

* DAVIS validation 영상의 과거 모델 선택 사용 여부를 확인하지 못해 탐색적 결과로 표시한다. 전체 지표·95% bootstrap CI·paired 비교·비용 표는 [summary.md](runs/run01/summary.md)와 [summary.json](runs/run01/summary.json)을 참조한다.

## 전환 시점 지표 (주 지표)

주 지표는 처리 프레임 `switch+1`..`switch+10`, 즉 Base+ 첫 10개 출력에서 annotation이 있는 프레임의 J&F 평균이다. `switch`는 Small의 마지막 출력이므로 채점하지 않는다. 평가 창 10은 `num_maskmem=7`과 별개의 설정이다. spatial memory와 object pointer는 서로 다른 history 범위를 사용한다.

- `switch_window`: 첫 10개 출력 중 annotation이 있는 프레임의 J/F/J&F 평균 및 채점 수
- `switch_frames`: `+1`부터 `+10`까지 각각의 처리 위치, 원본 stem, J/F/J&F, 채점 수와 상태
- `scored`: GT로 채점됨; `missing_annotation`: GT 없음; `outside_suffix`: 과거 짧은 결과의 suffix 밖
- 새 영상 평가에는 전환 후 10개 출력이 모두 있어야 한다. GT가 없는 위치도 기록하며 점수를 0으로 대체하지 않는다.
- Suffix J&F(`post_switch`)는 `switch+1`..`end` 전체 평균으로 함께 남긴다.

case별 score JSON과 `switch_frames.csv`에 프레임별 값을 저장한다. `summary.json`에는 `+1`..`+10`의 J/F/J&F video 평균·CI·coverage, `paired_switch_frames`, `paired_switch_window` 및 기존 suffix `paired`를 기록한다. `summary.md`에도 10개 offset을 모두 표로 표시한다. LVOS/VOST offset은 처리 프레임 단위이며 원본 기준 각각 5/6 프레임 간격이다.

기존 `runs/run01`, `runs/heldout01`의 공개 summary는 이전 suffix 기준 결과다. 원 workspace의 저장된 prediction cache에서 아래 명령으로 새 지표를 추가할 수 있다. 모델 load·추론은 하지 않으며, 기존 suffix 점수가 달라지면 중단한다. 새 translator 성능을 얻으려면 새 학습 후 새 run directory에서 추론해야 한다.

```sh
python test10/run.py --stage rescore --run-dir test10/runs/run01
python test10/run.py --stage rescore --run-dir test10/runs/heldout01
```

## 전환 직후·Base+ 일치도 진단

`handoff_diagnostics.py`는 저장된 prediction cache(`.pt`)만 읽어 두 지표를 계산한다. 모델 load·GPU 추론은 하지 않는다.

- Early J&F: 전환 후 +1..+K 프레임(기본 10, 프레임별 값도 별도 기록)의 GT J&F. 기존 `mvp_scoring.score` 경로를 그대로 사용한다.
- IoU vs Base+: GT 없이 Base+-native 예측 mask와의 프레임별 IoU. early 구간과 suffix 전체를 따로 보고하며, 전환 프레임에서 Small/Base+ mask가 일치한 case(`switch_iou ≥ 0.9`)만 따로 집계한다.

`.pt` cache와 원 workspace(`test9/`, `vos-data/`)가 있는 환경에서 원래 run directory를 대상으로 실행한다. 공개 사본은 cache가 없고 provenance 경로가 치환되어 있어 실행되지 않는다. 결과는 `<run-dir>/handoff_diagnostics.{json,md}`에 쓴다.

```sh
python test10/handoff_diagnostics.py --run-dir test10/runs/heldout01
python test10/handoff_diagnostics.py --run-dir test10/runs/run01
```

## 포함 파일과 큰 산출물

`runs/run01/artifacts/*.json`에는 case별 점수와 self-injection 결과가 있고 `.sha.json` sidecar는 content-addressed artifact의 hash를 보존한다. 크기가 큰 `.pt` 예측/state blob 1,924개(약 40.44 GiB)는 GitHub 저장소에 올리지 않았다. 따라서 점수·통계는 검토할 수 있지만, 이 공개 패키지만으로 binary mask/state를 복원하거나 해당 cache에서 `--resume`할 수는 없다.

`runs/heldout01`에도 선정 명단, case별 점수 JSON, gate, 시간, 요약 및 provenance를 포함했다. 약 38 GiB의 state·prediction `.pt` cache와 VOST 숫자형 symlink view는 포함하지 않았다. 공개용 manifest·provenance의 절대 경로는 `${WORKSPACE_ROOT}`로 치환했으므로, 이 사본에서 바로 `--resume`할 수 없다.

선택 manifest와 provenance의 로컬 절대 경로는 공개를 위해 `${WORKSPACE_ROOT}`로 치환했다. 입력 파일 SHA-256, checkpoint SHA-256, 모델·소프트웨어 정보는 유지했다. 원본 데이터, SAM 2 checkpoint, test9 translator 코드/checkpoint 및 600 fit shard는 별도 자산이며 이 폴더에 포함되지 않았다.

## 재현 관련 주의

기존 실험 실행 시 CPU 계약 테스트 22개가 통과했다. 실험은 SAM 2 + CUDA 환경에서 실행됐으며, 선택적 `_C` post-processing extension을 불러오지 못해 fill-holes 후처리를 건너뛴다는 upstream 경고가 기록됐다. 추론과 gate는 완료됐지만, 이 동작은 재현 환경에서 확인해야 한다.

Harness는 실행 시 sibling `test9/`, `vos-data/`, `vos-checkpoints/`, `sam2/`와 맞는 Python/CUDA 환경을 기대한다. 현재 GitHub 저장소 clone만으로는 이 실험을 독립 재실행할 수 없다. 따라서 `test9` 패키지가 없는 clone에서는 해당 패키지를 import하는 affine/model adapter 테스트도 실행되지 않는다(현재 repository-only 확인에서 `mvp_common` 부재로 실패). 원 실험 workspace에서는 기존 실험 실행 시 CPU 계약 테스트 22개가 통과했다. 필요한 데이터·checkpoint 권리와 경로를 준비한 뒤 실행한다. 이 실험 실행의 source 경로/hash와 설정은 [provenance.json](runs/run01/provenance.json)에 기록되어 있다.
