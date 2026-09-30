# test10 추가 영상 평가 결과

## 범위와 완료 상태

`runs/heldout01`은 기존 `runs/run01`과 겹치지 않는 영상 140개를 평가했다. MOSEv2 development의 미평가 영상, LVOS v2 validation, DAVIS 2017 train, VOST validation에서 각각 35개 영상을 선정했다. 영상마다 한 객체와 한 switch를 사용했으며, 30개씩의 기본 집합과 5개씩의 추가 집합을 추론 전 고정했다. 모델 성능을 보고 영상을 고르거나 checkpoint를 다시 학습하지 않았다.

Small-only, Base+-native, Direct Copy, Affine, Residual MLP, Spatial-context Transformer, Last-mask restart, Original-anchor+Replay-16을 같은 140 cases에서 평가했다. Base+ self-injection은 별도 정확성 gate였다. 기존 test10의 component ablation과 Replay-4/8은 이번 확대 평가에서 반복하지 않았다.

140/140 cases와 1,120/1,120 방법별 결과가 완료됐고, self-injection gate는 140/140 통과했다. 실패·중복 실행·평가 프레임 불일치는 없었다. 실제 평가 프레임은 MOSE 1,683, LVOS 7,968, DAVIS 1,219, VOST 1,977개로 모두 annotation이 있었다. 단일 RTX 4080 SUPER에서 smoke와 본 평가의 누적 실행 시간은 125.53분으로 3시간 예산 이내였다.

## 주요 결과

수치는 영상 단위 평균 J&F ×100이다. 두 번째 표의 대괄호는 영상을 재표본추출한 95% CI이다. 각 데이터셋의 표본 수는 독립 영상 35개다. 방법별 평균의 CI는 `runs/heldout01/summary.md`에 있다.

| 데이터셋 | Small-only | Base+-native | Direct | Affine | MLP | Transformer | Last-mask | Replay-16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MOSEv2 | 62.80 | 59.98 | 19.70 | 54.12 | 54.25 | 54.76 | 55.35 | 52.23 |
| LVOS v2 val | 76.24 | 76.29 | 7.17 | 73.09 | 73.28 | 74.48 | 69.92 | 60.67 |
| DAVIS 2017 train | 88.18 | 88.11 | 9.66 | 86.79 | 87.78 | 86.40 | 83.59 | 86.65 |
| VOST val | 47.08 | 45.04 | 6.74 | 46.27 | 42.89 | 44.85 | 45.61 | 39.92 |

| 데이터셋 | Affine − Direct | MLP − Affine | Transformer − Affine | Affine − Small-only |
|---|---:|---:|---:|---:|
| MOSEv2 | +34.42 [19.61, 49.16] | +0.12 [−2.30, 2.69] | +0.64 [−2.01, 3.81] | −8.68 [−15.45, −2.83] |
| LVOS v2 val | +65.92 [53.09, 77.61] | +0.19 [−2.33, 3.54] | +1.39 [−3.14, 7.75] | −3.15 [−8.45, 1.75] |
| DAVIS 2017 train | +77.13 [66.05, 86.31] | +0.99 [−0.03, 2.71] | −0.39 [−0.80, −0.03] | −1.38 [−3.52, 0.02] |
| VOST val | +39.53 [27.04, 50.87] | −3.39 [−9.88, 0.94] | −1.42 [−3.85, 0.49] | −0.80 [−3.83, 2.11] |

Affine은 모든 데이터셋에서 Direct Copy보다 크게 높았다. Nonlinear의 추가 이점은 일관되지 않았고, 35개 영상의 신뢰구간으로 작은 차이를 배제할 수도 없다. 새 MOSE 영상에서는 Affine이 Small-only보다 평균 8.68 point 낮아, 기존 집합의 “Small 품질을 대체로 유지한다”는 해석을 전체 MOSE로 일반화할 수 없다. LVOS validation의 Affine 73.09는 앞선 train development 집합의 94.26보다 낮다. 두 집합의 난도와 영상 분포가 다르므로 이를 같은 영상에서의 성능 하락량으로 해석하지 않는다.

DAVIS는 공식 train split을 사용했기 때문에 공식 held-out test 점수가 아닌 translator 기준의 보조 평가다. VOST는 물체 변형이 많은 별도 도메인이다. 이 평가는 각 데이터셋에서 객체별 독립 실행으로 얻은 test10 J&F이며, 다객체 동시 추적이나 VOST 공식 leaderboard 점수로 표기하지 않는다. 서로 다른 데이터셋의 점수는 합쳐서 하나의 주 결과로 제시하지 않는다.

## 재현 및 근거

- `runs/heldout01/selection.json`: 모든 영상, 객체, switch, RGB·annotation hash, split 및 기본/추가 집합.
- `runs/heldout01/provenance.json`: 평가 코드, SAM 2, translator checkpoint 및 환경 hash.
- `runs/heldout01/budget.json`, `schedule.json`, `extension_decision.json`: 실제 시도·시간과 점수를 사용하지 않은 추가 집합 결정.
- `runs/heldout01/summary.json`: J/F/J&F, paired video bootstrap CI 2,000회, visible·absent·reappearance 및 smoke 비용.
- `runs/heldout01/summary.md`: 데이터셋·방법별 간결한 표.

기존 평가 영상과의 중복 0개, MOSE·LVOS translator fit 영상과의 중복 0개, DAVIS 기존 validation 영상과의 중복 0개를 확인했다. 22개 CPU 계약 테스트와 `--report-only` provenance 검증도 통과했다. VOST의 `frameNNNNN` 파일명은 원본을 복사하지 않은 숫자형 symlink view로 읽었으며, 첫 프레임 객체 ID와 void label 255를 포함하는 기존 지표 경로를 사용했다.
