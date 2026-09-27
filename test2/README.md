# SAM 2.1 Small → Base+ memory preconditioning

 SAM 2.1의 Small과 Base+의 메모리 형태를 그대로 사용하지 않고 translate 이전에 약간의 전처리와 후처리를 통하여 변환 정확도와 속도를 높일 수 있는지 확인하는 테스트입니다. 

SAM 2.1 Small이 switch 시점까지 만든 객체별 temporal memory(`maskmem_features`, `obj_ptr`)를 고정 canonical transform과 작은 translator head로 변환해 Base+에 주입하고, Base+가 과거 프레임을 다시 처리하지 않고 `switch+1`부터 추적을 이어갈 수 있는지 검증하는 연구 코드입니다.

- 변환 후보: spatial `M0` raw, `M1` standardization, `M2` ZCA, `M3` output-projection SVD, `M4` consumer-aware / pointer `P0` raw, `P1` standardization, `P2` no-object anchor, `P2+P4` anchor+consumer, `P3` final-linear
- 학습: aligned pair 학습 → 같은 고정 transform을 유지한 native pair 미세조정. Transform 통계는 aligned fit split에서만 계산
- 최종 판단 기준: held-out native Small prefix → Base+ no-replay suffix J&F. State MSE와 K/V·attention 유사도는 진단 지표

자세한 프로토콜은 [docs/design/memory_preconditioning_protocol.md](docs/design/memory_preconditioning_protocol.md)에 있습니다.

## 현재 상태

- 완료: SAM 2.1 source/checkpoint audit, LVOS v2 train 80/10/10 source-group split, 전체 transform suite round-trip, LVOS 한 영상(`0ClBYzYm`)에 대한 `M0_raw`/`P0_raw` MLP fit sanity check
- 미완료: 후보 간 비교, VOST 학습, held-out no-replay continuation 평가. 현재 수치로는 방법에 대한 결론을 내리지 않습니다 ([report.md](experiments/sam21_small_to_baseplus_precondition/report.md))

코드 작성만 완료되고 실제로 테스트해보지는 않았습니다.
