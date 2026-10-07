# test10 — Small → Base+ Affine 상태 전환 실험

SAM 2.1 Small에서 Base+로 모델을 전환할 때, 선형 변환의 가능성을 검토해본다. 
간단한 선형 구조 Wx+b 를 통하여 상대적으로 유사한 small 과 base+의 representation 차이가
단순한 선형(affine?) 변환으로 설명될 수 있는지 확인한다. 

선형 모델 학습 조건은 다음과 같다. 학습 조건 및 비교군은 최대한 non-linear translator의 
transfomer과 유사하게 설정하였다.

maskmem_features에 대해서는
$$
\hat{M}_{h,w} = M_{h,w}W_s + b_s
$4
obj_ptr에 대해서는 
$$
\hat{p} = pW_p + b_p
$$
로 각각 선형 변환을 수행한다


### Affine 모델 학습 조건

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


### 중간 실험 결과
##### 전환 후 +1~+10 J&F 변화

Affine의 J&F 결과입니다.  
영상 내 여러 pair의 결과를 먼저 평균한 뒤, 각 영상을 동일한 가중치로 평균했습니다.

| 프레임 | MOSEv2 | LVOS 학습 영상 | LVOS 선택 영상 |
|---|---:|---:|---:|
| +1 | 77.75 | 95.42 | 95.87 |
| +2 | 77.42 | 95.24 | 95.88 |
| +3 | 76.74 | 95.30 | 96.33 |
| +4 | 76.97 | 95.17 | 95.18 |
| +5 | 76.68 | 95.12 | 95.61 |
| +6 | 77.24 | 95.32 | 94.86 |
| +7 | 76.96 | 95.25 | 95.30 |
| +8 | 76.85 | 95.04 | 95.01 |
| +9 | 76.07 | 95.49 | 95.71 |
| +10 | 75.81 | 95.26 | 95.61 |

###### 전체 후속 구간 J&F

| 데이터 | 완료 pair | Direct Copy | Affine | 저장된 Base+ 상태 기준 |
|---|---:|---:|---:|---:|
| MOSE | 1,694 | 27.50 | **71.95** | 73.22 |
| LVOS 학습 | 1,352 | 3.38 | **93.83** | 95.15 |
| LVOS 선택 | 296 | 3.69 | **93.89** | 95.03 |


<img width="2400" height="720" alt="image" src="https://github.com/user-attachments/assets/2f5b3ee8-6751-4f5c-980d-f5f59b2a5311" />
