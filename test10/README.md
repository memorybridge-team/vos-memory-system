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
