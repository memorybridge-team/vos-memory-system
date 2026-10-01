# Affine: 변환 방식과 전환 직후·전체 구간 비교

추가 추론이나 학습 없이 기존 프레임 점수를 다시 집계했다. Affine은 Direct Copy의 성능 붕괴를 크게 줄이며 초기 5프레임에서 nonlinear 모델과 대체로 비슷하다. 전체 suffix에서는 데이터셋에 따라 MLP의 이점이 나타나지만, Transformer의 일관된 추가 이점과 Affine의 일률적인 시간적 악화는 확인되지 않았다.

## 실제 변환과 학습

Spatial memory의 각 64차원 위치 벡터에 `m_hat[k,h,w] = W_m @ m[k,h,w] + b_m`을 적용한다. `W_m`은 64×64, `b_m`은 64다. Object pointer에는 독립적인 `p_hat[k] = W_p @ p[k] + b_p`를 적용한다. `W_p`는 256×256, `b_p`는 256이다. 총 69,952개 파라미터다.

모든 memory record·위치에 같은 가중치를 적용한다. 공간·시간 혼합, 활성화 함수, attention은 없다. Spatial head는 1×1 convolution과 동등하다. Target PE는 Base+에서 생성하며 과거 source mask·score는 주입하지 않는다. Handoff 시 한 번 적용하고 과거 RGB replay 없이 Base+가 다음 프레임부터 처리한다.

현재 checkpoint는 기존 epoch-30 가중치가 아니다. MOSE·LVOS의 학습 1,000 pairs/750영상, validation 200 pairs/175영상을 사용해 새로 학습했다. 학습의 유효 memory record는 15,189개다. `W=I,b=0`으로 시작해 **Adam**, lr 0.001, seed 7, FP32, 8 epochs로 학습했다. `spatial MSE + pointer MSE`이며 SAM 모델을 업데이트하지 않는다. `batch_records=4`는 record microbatch이고 pair 전체 gradient를 모아 pair당 한 번 optimizer step을 수행한다. Validation은 영상별 균등 state loss다. Affine은 epoch 6/6,000 updates, validation loss 0.243274인 checkpoint를 선택했다. MLP·Transformer는 epoch 8이 선택되어 수렴 여부는 확인되지 않았다.

## 공통 비교 집합

MOSE 40, LVOS 39, DAVIS 40, VOST 40의 **159개 영상·159 cases**, prompt index=0, 영상당 객체 하나를 독립적으로 추적한다. 학습·checkpoint 선택 영상과 분리됐지만 대부분 이전 평가 이력이 있다. MOSE local train/development, LVOS valid, DAVIS train exploratory, VOST val이다. 공식 최종 benchmark 결과가 아니다.

GT-visible frame만 평균하고 영상별 평균을 균등 집계했다. Void label 255는 제외했다. VOST는 J, 나머지는 J&F다. 표는 0..100 point이며 서로 다른 데이터셋 열을 단순히 하나의 품질 점수로 섞지 않는다. Full Replay는 native Base+ 품질 참조이지 반드시 최대 품질을 내는 절대 상한선은 아니다.

## 전환 직후 +1..+5 평균

| 방법 | MOSE J&F | LVOS J&F | DAVIS J&F | VOST J |
|---|---:|---:|---:|---:|
| Small-only | 64.72 | 73.96 | 85.77 | 56.40 |
| Base+-native / Full Replay | 60.41 | 76.98 | 87.76 | 50.11 |
| Direct Copy | 0.68 | 0.26 | 3.63 | 0.06 |
| Affine | 59.11 | 75.64 | 83.96 | 56.03 |
| Residual MLP | 57.11 | 75.56 | 85.09 | 53.56 |
| Transformer | 60.22 | 75.48 | 84.60 | 54.59 |
| Last Mask | 53.33 | 71.88 | 81.62 | 56.08 |
| Anchor + Replay-16 | 51.08 | 50.32 | 86.14 | 40.29 |

적어도 한 visible frame이 있는 영상 수는 MOSE 33, LVOS 36, DAVIS 40, VOST 36이며 모든 방법의 평가 support는 동일하다. Affine−Direct는 각각 +58.43/+75.38/+80.33/+55.97점이다.

### +1, +2, +3, +4, +5의 순차 결과

| 데이터셋 | 방법 | +1 | +2 | +3 | +4 | +5 |
|---|---|---:|---:|---:|---:|---:|
| MOSE | Base+-native | 56.23 | 62.45 | 61.56 | 60.54 | 64.81 |
| MOSE | Affine | 59.98 | 61.71 | 61.15 | 58.74 | 63.73 |
| MOSE | MLP | 58.29 | 61.58 | 57.57 | 57.35 | 58.22 |
| MOSE | Transformer | 59.70 | 64.10 | 64.15 | 59.16 | 63.96 |
| LVOS | Base+-native | 74.99 | 76.79 | 77.40 | 77.57 | 78.14 |
| LVOS | Affine | 73.16 | 75.84 | 76.58 | 76.12 | 76.51 |
| LVOS | MLP | 73.19 | 75.70 | 76.49 | 76.01 | 76.40 |
| LVOS | Transformer | 72.72 | 75.37 | 76.65 | 76.14 | 76.53 |
| DAVIS | Base+-native | 86.93 | 86.88 | 86.80 | 89.21 | 88.92 |
| DAVIS | Affine | 84.10 | 83.98 | 86.06 | 85.61 | 86.49 |
| DAVIS | MLP | 84.09 | 84.12 | 86.46 | 88.40 | 88.98 |
| DAVIS | Transformer | 83.69 | 83.79 | 86.16 | 87.66 | 88.27 |
| VOST | Base+-native | 51.30 | 50.55 | 49.80 | 50.02 | 50.07 |
| VOST | Affine | 56.81 | 57.33 | 55.05 | 54.73 | 57.63 |
| VOST | MLP | 54.20 | 54.30 | 52.65 | 51.84 | 56.18 |
| VOST | Transformer | 54.85 | 54.89 | 52.77 | 54.63 | 57.19 |

Offset는 처리 프레임 기준이다. LVOS raw stride=5, VOST=6이다. 가시성에 따라 각 offset의 영상 수가 달라질 수 있어 다섯 열의 단순 평균과 first5 영상별 집계가 반드시 같지는 않다.

## 추론 완료 후 전체 suffix

마지막 프레임 한 장이 아니라 switch 이후 끝까지의 visible frame 평균이다. 영상 수는 MOSE/LVOS/DAVIS/VOST=40/39/40/40이다.

| 방법 | MOSE J&F | LVOS J&F | DAVIS J&F | VOST J |
|---|---:|---:|---:|---:|
| Small-only | 58.00 | 76.36 | 86.46 | 46.79 |
| Base+-native / Full Replay | 54.84 | 75.41 | 86.92 | 39.77 |
| Direct Copy | 0.07 | 0.02 | 2.76 | 0.00 |
| Affine | 48.55 | 68.61 | 83.89 | 43.59 |
| Residual MLP | 50.20 | 72.50 | 85.28 | 43.34 |
| Transformer | 48.79 | 69.25 | 84.67 | 43.12 |
| Last Mask | 44.68 | 62.48 | 80.67 | 43.63 |
| Anchor + Replay-16 | 47.93 | 57.84 | 85.64 | 34.14 |

MLP−Affine는 MOSE +1.65, LVOS +3.89, DAVIS +1.39, VOST −0.25점이다. DAVIS의 paired 95% CI는 +0.19..+3.33점이며 LVOS는 −0.40..+9.87점이다. Transformer−Affine는 +0.24/+0.64/+0.77/−0.47점이고 네 데이터셋의 CI가 모두 0을 포함한다. Paired video bootstrap 5,000 draws, seed 7이며 다중 비교 보정하지 않은 pointwise CI다.

Affine은 전체 suffix에서 Small-only보다 네 데이터셋 모두 낮다. 표현 변환이 Direct Copy보다 낫다는 결과와 Base+로 전환할 가치가 있다는 결과는 서로 다른 주장이다.

### 마지막 프레임만 비교

마지막 프레임에서 visible인 영상 수는 36/36/37/35다.

| 방법 | MOSE | LVOS | DAVIS | VOST |
|---|---:|---:|---:|---:|
| Base+-native | 56.41 | 72.81 | 88.20 | 34.52 |
| Affine | 43.88 | 65.59 | 83.77 | 35.99 |
| MLP | 46.40 | 68.81 | 84.91 | 36.47 |
| Transformer | 40.61 | 65.48 | 84.09 | 38.00 |

Last5 및 모든 baseline의 상세 값은 `windows.csv`에 보존했다. 단일 마지막 프레임은 전체 구간 결론의 보조 지표다.

## 장기 악화 및 큰 bank에 관한 결론

같은 영상에서 `native−Affine` gap의 full-suffix−first5 변화를 계산하면 MOSE +4.32 [−1.08,10.48], LVOS +2.84 [−0.85,7.34], DAVIS −0.77 [−3.42,0.85], VOST −0.13 [−4.25,3.81]점이다. 모두 0을 포함한다. 일부 데이터셋의 평균 경향은 있으나 Affine이 시간에 따라 일률적으로 악화된다고 확정할 수 없다.

별도 다운로드 bank의 학습·선택 분리 집합에서도 초기 5프레임의 Affine/Transformer는 MOSE 67.87/68.14 J&F, LVOS 95.04/95.09 J&F로 가깝다. 이 집합은 5프레임만 있고 native/source 참조와 장기 결과가 없어 159-case 결과와 합치지 않는다.

표현 관계의 상당 부분을 채널 정렬로 해결할 수 있다는 것이 가능한 해석이다. 완전한 선형 관계의 증명은 아니다. Affine은 Small이 잃은 정보를 복원한다고 보장하지 않으며 state MSE 최소화도 장기 추적 품질 최대화와 같지 않다. 현재 근거는 **강력한 저복잡도 baseline**이라는 결론을 지지하지만 모든 데이터셋의 최선 방법이라는 결론은 지지하지 않는다.

원본 근거: [windows.csv](windows.csv), [affine_differences.csv](affine_differences.csv), [metrics.json](metrics.json).
