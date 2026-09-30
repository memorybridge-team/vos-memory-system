# test10 결과

완료 140 / 후보 140 cases. 재사용 row 0.

모든 방법 및 self-injection gate를 완료한 공통 case만 집계합니다. 부분 실행은 summary.json의 incomplete에 보존합니다.
heldout DAVIS uses train split and is exploratory

| Dataset | Method | Cases | Videos | J&F ×100 (95% CI) |
|---|---|---:|---:|---|
| MOSEv2 | small_only | 35 | 35 | 62.80 [50.96, 73.44] |
| MOSEv2 | base_native | 35 | 35 | 59.98 [48.43, 70.99] |
| MOSEv2 | direct | 35 | 35 | 19.70 [12.38, 27.78] |
| MOSEv2 | affine | 35 | 35 | 54.12 [42.58, 66.06] |
| MOSEv2 | residual_mlp | 35 | 35 | 54.25 [42.91, 65.55] |
| MOSEv2 | transformer | 35 | 35 | 54.76 [43.32, 66.35] |
| MOSEv2 | last_mask | 35 | 35 | 55.35 [43.62, 66.89] |
| MOSEv2 | anchor_replay_16 | 35 | 35 | 52.23 [40.18, 63.92] |
| LVOSv2 | small_only | 35 | 35 | 76.24 [66.22, 85.29] |
| LVOSv2 | base_native | 35 | 35 | 76.29 [64.96, 86.68] |
| LVOSv2 | direct | 35 | 35 | 7.17 [3.19, 12.16] |
| LVOSv2 | affine | 35 | 35 | 73.09 [61.59, 83.48] |
| LVOSv2 | residual_mlp | 35 | 35 | 73.28 [61.59, 84.00] |
| LVOSv2 | transformer | 35 | 35 | 74.48 [63.95, 84.47] |
| LVOSv2 | last_mask | 35 | 35 | 69.92 [58.39, 80.68] |
| LVOSv2 | anchor_replay_16 | 35 | 35 | 60.67 [47.04, 74.07] |
| DAVIS2017 | small_only | 35 | 35 | 88.18 [82.79, 92.49] |
| DAVIS2017 | base_native | 35 | 35 | 88.11 [82.49, 92.53] |
| DAVIS2017 | direct | 35 | 35 | 9.66 [3.71, 17.35] |
| DAVIS2017 | affine | 35 | 35 | 86.79 [80.84, 91.49] |
| DAVIS2017 | residual_mlp | 35 | 35 | 87.78 [82.30, 92.12] |
| DAVIS2017 | transformer | 35 | 35 | 86.40 [80.41, 91.23] |
| DAVIS2017 | last_mask | 35 | 35 | 83.59 [74.41, 90.95] |
| DAVIS2017 | anchor_replay_16 | 35 | 35 | 86.65 [80.94, 91.54] |
| VOST | small_only | 35 | 35 | 47.08 [36.19, 57.56] |
| VOST | base_native | 35 | 35 | 45.04 [35.67, 54.13] |
| VOST | direct | 35 | 35 | 6.74 [2.04, 12.46] |
| VOST | affine | 35 | 35 | 46.27 [35.08, 56.65] |
| VOST | residual_mlp | 35 | 35 | 42.89 [32.14, 53.87] |
| VOST | transformer | 35 | 35 | 44.85 [34.31, 54.61] |
| VOST | last_mask | 35 | 35 | 45.61 [34.15, 56.56] |
| VOST | anchor_replay_16 | 35 | 35 | 39.92 [29.88, 49.86] |

세부 cohort, component ablation, visible/absent/reappearance, paired CI 및 비용은 summary.json을 참조하세요.
