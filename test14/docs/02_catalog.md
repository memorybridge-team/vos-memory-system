# 02 방법과 실행 목록

## 평가 방식

| 항목 | 내용 |
|---|---|
| J&F | (J+F)/2, 전환 뒤 객체가 보이는 프레임만 채점 |
| 집계 | 프레임 → 케이스 → 영상 → 전체 평균 |
| 차이 | 같은 케이스끼리 짝지은 값 |
| 95% 신뢰구간 | 영상 단위 부트스트랩 20,000회, seed 7 |
| MOSE dev, LVOS val, LVOS dev | 저장된 state pair로 전환하고 Base+로 이어 추적 (`handoff_eval`, `jf_eval`, `more_eval`) |
| M3-VOS, PUMaVOS | Small을 실제로 돌려 전환 (`unseen_eval`, `unseen_more`) |

표의 `better_by_1pt`, `worse_by_1pt`는 비교 기준보다 1점 넘게 높거나 낮은 영상 수이다.

## 번역기 이름

| 이름 | 내용 |
|---|---|
| `legacy` | 1차 공식 학습 transformer, 에폭 27. 논문 모델이 아니다 |
| 논문 Translator | `r2` 학습 transformer, 에폭 26. 이번 실험에서 다시 돌리지 않았다 |
| `ridge` | 닫힌 형식 선형 번역기 |
| `mlp`, `transformer`, `mlp_context` | 이번 실험의 MSE 학습 모델. 코드상 `residual_mlp`, `transformer`, `mlp_context` 구조 |
| `ds`, `ds2`, `ds3`, `ds_context`, `conv3lin`, `conv3mlp`, `mlp2`, `mlp512`, `affine`, `transformer_ridge` | 공간 구조 후보 |
| `e2e_ctx`, `e2e_mlp`, `e2e_tf` | 2라운드 파인튜닝 |
| `e2e2_ctx` | Dice와 object score 가중치를 키운 변형 |
| `e2e3_ctx`, `e2e3_mlp`, `e2e3_tf`, `e2e3_dsc` | 3라운드 파인튜닝 |

결과 표에서 뒤에 붙는 `_f`는 마지막 가중치라는 뜻이다. `e2e_ctx_f`는 `weights/spatial/e2e_ctx_last`, `e2e_mlp_f`는 `e2e_mlp_last`, `e2e_tf_f`는 `e2e_tf`, `e2e3_*_f`는 해당 폴더의 `model.pt`이다.

이번 실험의 학습 모델은 모두 선형 ridge 포인터(65,792개)를 쓴다.

## 방법 이름

전체 정의는 [pipeline 스크립트](../code/review)의 `--method`에 있다. 형식 `a,b,c,d`는 조건부 공간, 나머지 공간, 조건부 포인터, 나머지 포인터의 출처이다.

| 이름 | 뜻 |
|---|---|
| `target_ref` | Base+ 실제 상태를 주입 |
| `base` | 외부 데이터셋에서 Base+를 처음부터 실행 |
| `small` | 전환하지 않고 Small 유지 |
| `direct` | Small 상태를 변환 없이 복사 |
| `*_nc` | 조건부 기록은 Base+ 값, 나머지만 해당 번역기 |
| `orig`, `lastvis`, `olv` | 프롬프트만, 마지막 마스크만, 둘 다 |
| `olv_soft` | 마지막 마스크를 만료되는 최근 기록으로 넣은 OLV |
| `replay4/8/16` | 프롬프트 + 전환 직전 k프레임 재처리 |
| `mm` | Moment-Matched Copy |
| `ridge_anchor`, `ridge_nc_anchor`, `recent_ridge` | 앵커와 번역 기록의 조합 |
| `pinb`, `pinp` | Base+가 전환 프레임에서 찾은 마스크를 앵커로 고정 |
| `g110~g150`, `ridge_gain`, `m110_ctx` | 평균 기준 출력 대비를 1.1~1.5배 |
| `swapK` | K프레임 뒤 번역 조건부 기록을 Base+ 기록으로 교체 |
| `pinK`, `*_pinK` | K프레임 뒤 Base+ 기록을 조건부 칸에 추가. 번역 기록은 유지 |
| `cond25`, `cond50` | 조건부 공간 오차를 25%, 50%로 축소 |
| `cond_noise`, `recent_noise` | 같은 크기의 무구조 잡음 |
| `cond_err`, `err_near/far/low/high/shrink/rest` | 조건부 공간 오차와 그 분해 성분만 Base+ 상태에 주입 |
| `legacy_sp_only`, `legacy_ptr_only`, `*_oracle`, `legacy_meanptr` | 오차 성분별 민감도 |
| `e2e_ctx_fc` | 파인튜닝 모델을 조건부 기록에만 적용 |
| `e2e_ctx3k`, `e2e_ctx3k_c` | 3,000스텝 중간 모델 |

## 실행 태그

원자료는 `results/raw/jf/<태그>.tar.gz`, `results/raw/unseen/<태그>.tar.gz`이다. `_mose`는 MOSE dev, `_val`은 LVOS val, `_lvd`는 LVOS dev이다.

| 태그 | 케이스 | 내용 | 스크립트 |
|---|---:|---|---|
| `sens` | 300 / 140 | 오차 민감도 | `run_sensitivity.sh` |
| `cand` | 10 / 126 | 조건부 재인코딩 후 구조 후보. 파인튜닝으로 의미가 줄어 중단 | `run_queue.sh` |
| `t1` | 300 / 140 | 비교군과 앵커 | `pipeline2.sh` |
| `t2` | 300 / 140 | 전체 번역 모델 비교, 조건부 오차 축소 | `pipeline2.sh` |
| `t4` | 300 / 140 | Replay, Moment-Matched, Base+ 앵커 | `pipeline3.sh` |
| `t5` | 300 / 140 | 이득, 교체, 고정 | `pipeline4.sh` |
| `t6` | 300 / 138 | 오차 분해 | `pipeline4.sh` |
| `t7` | 1 | 이득 후보. 파인튜닝이 대비를 복원해 대부분 뺌 | `pipeline5.sh` |
| `t9` | 300 | 3,000스텝 중간 모델 | `pipeline6.sh` |
| `t10` | 300 / 140 | `e2e_ctx` 마지막 가중치, 7프레임 고정 | `pipeline8.sh`, `pipeline10.sh` |
| `t11` | 300 | `e2e_mlp`, `e2e2_ctx` | `pipeline8.sh`, `pipeline11.sh` |
| `t12` | 300 | 3라운드 | `pipeline9.sh` |
| `t13`, `t15` | 300 | 고정 시점 1~30프레임 | `pipeline11.sh`, `pipeline13.sh` |
| `t14`, `t16` | 73 | LVOS dev 후보 선택 | `pipeline13.sh`, `pipeline14.sh` |
| `t17` | 140 | 최종 후보 LVOS val | `pipeline16.sh` |
| `t18` | 300 / 140 | `e2e_tf` | `pipeline15.sh` |
| `t19` | 300 / 140 / 73 | `e2e3_mlp`, `e2e3_tf` | `pipeline17.sh` |
| `t20` | 300 / 73 | 학습 영상 J&F | `pipeline18.sh` |
| `fp32` | 30 / 10 | 정밀도 영향. 중단으로 일부만 완료 | `pipeline2.sh` |
| `pumavos`, `m3vos` | 26 / 110 | 외부 데이터셋 1차 | `run_unseen.sh` |
| `pumavos2`, `m3vos2` | 26 / 79 | 외부 비교군. `m3vos2`는 중단으로 일부만 완료 | `pipeline2.sh` |
| `pumavos3`, `m3vos3` | 26 / 110 | 2라운드와 7프레임 고정 | `pipeline12.sh` |
| `pumavos4`, `m3vos4` | 26 / 110 | 3라운드와 5프레임 고정 | `pipeline16.sh` |
| `pumavos5`, `m3vos5` | 26 / 110 | `e2e_mlp`, `e2e_tf` | `pipeline19.sh` |
| `smoke`, `diag1`, `diag2` | 소수 | 동작 확인과 진단 | – |

한 방법이 여러 태그에 있으면 `build_tables.py`의 `PRIORITY` 순서로 하나를 쓴다. `jf_case_scores.csv`에는 모든 태그의 행이 그대로 있다.

## 표

`results/tables`의 CSV이다.

| 파일 | 내용 |
|---|---|
| `jf_case_scores.csv` | MOSE, LVOS의 태그·케이스·방법별 점수 |
| `jf_levels.csv`, `jf_paired.csv` | 그룹별 수준값과 짝비교. `videos_scope`의 `clean`은 같은 장면 영상을 뺀 값 |
| `jf_windows.csv` | 전환 뒤 구간별 값 |
| `external_case_scores.csv`, `external_levels.csv`, `external_paired.csv` | M3-VOS, PUMaVOS |
| `paper_*.csv` | 논문 평가 DB에서 뽑은 값과 이번 실험과의 비교 |
| `training_models.csv`, `training_e2e_history.csv`, `pointer_experiments.csv` | 학습 설정과 경과 |
| `cost_summary.csv` | 전환 비용 |
| `scene_overlap.csv`, `scene_overlap_videos.csv` | 장면 겹침 |
