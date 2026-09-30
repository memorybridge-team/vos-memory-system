# test10 결과

완료 148 / 후보 758 cases. 재사용 row 0.

모든 방법 및 self-injection gate를 완료한 공통 case만 집계합니다. 부분 실행은 summary.json의 incomplete에 보존합니다.
DAVIS는 과거 모델 선택 사용 여부가 확인되지 않아 탐색적 평가로 표시합니다.

| Dataset | Method | Cases | Videos | J&F ×100 (95% CI) |
|---|---|---:|---:|---|
| MOSEv2 | small_only | 77 | 77 | 63.99 [56.00, 71.32] |
| MOSEv2 | base_native | 77 | 77 | 71.78 [64.39, 78.58] |
| MOSEv2 | direct | 77 | 77 | 31.42 [23.14, 40.13] |
| MOSEv2 | affine | 77 | 77 | 64.85 [56.78, 71.96] |
| MOSEv2 | affine_spatial | 77 | 77 | 64.01 [55.43, 71.39] |
| MOSEv2 | affine_pointer | 77 | 77 | 31.47 [23.25, 40.16] |
| MOSEv2 | residual_mlp | 77 | 77 | 60.12 [51.65, 67.53] |
| MOSEv2 | transformer | 77 | 77 | 64.90 [56.58, 72.28] |
| MOSEv2 | last_mask | 77 | 77 | 60.69 [52.24, 68.00] |
| MOSEv2 | anchor_replay_4 | 77 | 77 | 56.41 [47.76, 64.58] |
| MOSEv2 | anchor_replay_8 | 77 | 77 | 60.61 [52.07, 68.57] |
| MOSEv2 | anchor_replay_16 | 77 | 77 | 65.98 [57.99, 73.05] |
| LVOSv2 | small_only | 41 | 40 | 95.57 [94.53, 96.57] |
| LVOSv2 | base_native | 41 | 40 | 95.89 [94.82, 96.78] |
| LVOSv2 | direct | 41 | 40 | 3.64 [0.54, 9.38] |
| LVOSv2 | affine | 41 | 40 | 94.26 [90.55, 96.53] |
| LVOSv2 | affine_spatial | 41 | 40 | 94.26 [90.57, 96.51] |
| LVOSv2 | affine_pointer | 41 | 40 | 3.65 [0.54, 9.38] |
| LVOSv2 | residual_mlp | 41 | 40 | 95.24 [93.64, 96.55] |
| LVOSv2 | transformer | 41 | 40 | 95.21 [93.60, 96.53] |
| LVOSv2 | last_mask | 41 | 40 | 95.01 [93.26, 96.44] |
| LVOSv2 | anchor_replay_4 | 41 | 40 | 84.77 [74.76, 93.51] |
| LVOSv2 | anchor_replay_8 | 41 | 40 | 84.86 [74.86, 93.62] |
| LVOSv2 | anchor_replay_16 | 41 | 40 | 86.72 [77.06, 94.20] |
| DAVIS2017 | small_only | 30 | 30 | 89.56 [83.58, 93.97] |
| DAVIS2017 | base_native | 30 | 30 | 90.26 [84.30, 94.48] |
| DAVIS2017 | direct | 30 | 30 | 10.42 [2.74, 20.44] |
| DAVIS2017 | affine | 30 | 30 | 90.02 [84.44, 94.25] |
| DAVIS2017 | affine_spatial | 30 | 30 | 90.00 [84.41, 94.21] |
| DAVIS2017 | affine_pointer | 30 | 30 | 4.88 [0.87, 10.87] |
| DAVIS2017 | residual_mlp | 30 | 30 | 90.08 [84.14, 94.45] |
| DAVIS2017 | transformer | 30 | 30 | 89.97 [84.32, 94.19] |
| DAVIS2017 | last_mask | 30 | 30 | 90.24 [84.32, 94.46] |
| DAVIS2017 | anchor_replay_4 | 30 | 30 | 78.63 [66.69, 88.15] |
| DAVIS2017 | anchor_replay_8 | 30 | 30 | 84.68 [75.33, 92.39] |
| DAVIS2017 | anchor_replay_16 | 30 | 30 | 84.61 [75.25, 92.30] |

세부 cohort, component ablation, visible/absent/reappearance, paired CI 및 비용은 summary.json을 참조하세요.
