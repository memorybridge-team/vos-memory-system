# Nonlinear MLP 첫 internal validation

## 한 줄 결론

SAM 2.1 Tiny의 기억을 Large용으로 바꾸는 Nonlinear Residual MLP는 Direct Copy보다
일부 추적 성능을 개선했지만, 아직 Target-Native 수준과는 큰 차이가 있으며
object presence 처리와 영상 간 일반화가 다음 병목이다.

## 무엇을 비교했는가

- Source/Target: SAM 2.1 Tiny → Large
- Translator: component-wise Residual MLP, hidden dimension 128, 82,498 parameters
- 학습: DAVIS 2017 train의 12개 영상, 29개 handoff case
- 내부 검증: 학습 영상과 겹치지 않는 3개 영상, 8개 handoff case
- 평가 가능한 visible case: 5개
- 전환 뒤 GT가 계속 비어 있는 absent-only case: 3개
- Seed: 7
- Epochs: 120
- Learning rate: 0.001
- Spatial samples: state pair당 4,096개

이 평가는 DAVIS 공식 benchmark 점수가 아니라 DAVIS train 내부의 video-level
validation이다.

## 결과

| 지표 | Direct Copy | Nonlinear MLP | 해석 |
|---|---:|---:|---|
| GT-visible J&F | 0.014222 | 0.112462 | MLP가 +0.098240 개선 |
| Target-Native GT-visible J&F | 0.653399 | 0.653399 | 두 방법이 따라가야 할 참고값 |
| 첫 5개 visible frame switch shock | 0.451226 | 0.464767 | 낮을수록 좋으며 MLP가 0.013541 악화 |
| Identity-break proxy rate | 1.000 | 0.800 | MLP가 20%p 개선했지만 여전히 높음 |
| Recovery rate | 0.000 | 0.200 | MLP 한 case만 회복 |
| GT-absent-only J&F | 1.000000 | 0.807692 | MLP가 빈 장면에서 false positive를 늘림 |

### Case별 GT-visible J&F

| Sequence / object / switch | Direct | Nonlinear MLP | Target-Native |
|---|---:|---:|---:|
| lady-running / 2 / 11 | 0.035548 | 0.000000 | 0.695507 |
| lady-running / 2 / 12 | 0.035562 | 0.000000 | 0.695507 |
| lindy-hop / 4 / 51 | 0.000000 | 0.000000 | 0.669817 |
| miami-surf / 2 / 6 | 0.000000 | 0.029639 | 0.603082 |
| miami-surf / 2 / 8 | 0.000000 | 0.532671 | 0.603082 |

## 해석

MLP가 `miami-surf / object 2 / switch 8`에서는 Target-Native에 가까운 J&F를
회복했다. 따라서 source state에 유용한 정보가 전혀 없거나 Nonlinear mapping이
항상 무효인 것은 아니다. 반면 다른 네 visible case에서는 거의 실패해 현재 모델의
영상 간 일반화가 부족하다.

Offline state 학습 loss는 27.229357에서 16.117426으로 감소했고 validation cosine은
spatial memory 0.765572, object pointer 0.727052였다. 하지만 presence-logit MSE가
11.569884로 컸고, absent-only 성능도 Direct보다 낮았다. Tensor alignment 개선이
downstream mask 성능을 자동으로 보장하지 않는다는 프로젝트의 핵심 가정을 다시
확인했다.

## 다음 실험

1. 학습된 MLP의 spatial memory와 object pointer는 유지하고 presence만 Direct Copy로
   교체해 presence 병목을 분리한다.
2. spatial-only, spatial+pointer, full-state ablation으로 어느 component가 실제 mask를
   개선하거나 망치는지 확인한다.
3. component별 loss scale을 정규화하고 validation 기준 early stopping을 추가한다.
4. 위 변경 뒤에도 대부분의 video에서 실패하면 더 많은 학습 영상과 event-balanced
   sampling, slot/context interaction 구조를 검토한다.

## 재현성과 무결성

- Code commit: `6530737`
- SAM 2 commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Selection manifest SHA-256:
  `aa5591bef9ab7f14cf8b3003a58f0f51d2e2c441497ef138a51cb4107b5340ce`
- Translator artifact SHA-256:
  `4e7541dd66dfd6d077ef026ece9b55eeb0bda34c96fc3544012a8efd176653d6`
- Training report SHA-256:
  `1b990d6453b5f994db7f22634918cc8fcb2b08ffebff37ec4bab09e618aeb3b5`
- RunPod test: `42 passed`

