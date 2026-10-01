# 2026-10-02 결과 공개: 기존 파일을 보존한 test10 산출물

이번 업로드는 앞선 실험과 2026-10-01자 분석 파일들을 함께 공개하는 작업이다. 기존 CSV·JSON·그래프·학습 기록의 바이트를 수정하지 않으며, 새 추론·학습·benchmark 제출을 수행하지 않았다. 날짜가 포함된 기존 폴더명도 유지한다.

## 결론을 확인하는 순서

1. [Affine 구간별 비교 보고서](affine_comparison_20261001/REPORT.md): 변환 방식, 실제 학습, 전환 직후 +1..+5, 전체 suffix와 마지막 프레임 비교.
2. [구간별 J/F/J&F](affine_comparison_20261001/windows.csv), [영상 단위 paired 차이·95% CI](affine_comparison_20261001/affine_differences.csv), [원본 집계 JSON](affine_comparison_20261001/metrics.json).
3. [논문 지표 집계 JSON](paper_metrics_20261001/metrics.json): 지표 정의, 데이터 출처, 한계, 미완료 항목, 원시 마스크·annotation의 SHA-256.
4. [품질·실패·ID swap](paper_metrics_20261001/quality.csv), [case별 회복률](paper_metrics_20261001/case_recovery.csv), [비용](paper_metrics_20261001/costs.csv), [drift](paper_metrics_20261001/drift.csv), [drift 그래프](paper_metrics_20261001/drift.svg).
5. [프레임별 점수](paper_metrics_20261001/visible_frame_scores.csv): 이름과 달리 annotation이 있는 absent 프레임도 포함한다. GT-visible 집계에는 `gt_present=True`만 사용한다. J/F/J&F 열은 0..1, 구간 집계 CSV는 0..100이다.
6. [중단한 대규모 pair 평가](paper_metrics_20261001/stopped_bank_first5.csv): 학습·모델 선택 영상을 제외한 초기 5프레임 분석. Native160/159와 섞지 않는다.

## 보존한 결과 묶음

| 폴더 | 내용 | 해석 |
|---|---|---|
| `fit1000_eval160` | 학습·검증 pair 목록, 160-case manifest, 8개 방법 case 점수, +1..+10, full suffix, gates, costs, audit, provenance, 학습 곡선 | 원래 실행 export를 그대로 보존 |
| `requested_metrics_20261001` | 초기 요청 지표 집계, native case 점수, 대규모 pair 점수 출처 | 과거 정의의 집계이며 최종 회복률은 아래 paper 집계 사용 |
| `paper_metrics_20261001` | GT-visible, void 제외, case별 회복률·격차 회복률, failure, ID swap, 출력 일치도, drift, 비용 | 논문 요청 지표에 맞춘 CPU 재집계. 아직 완전한 최종 benchmark가 아님 |
| `affine_comparison_20261001` | prompt=0인 공통 159개 영상의 +1..+5, first5, full suffix, last5, last frame 및 paired CI | 최신 Affine 비교의 근거 |
| `publication_20261002` | 원시 점수·실행 metadata·로그·외부 코드 snapshot의 무손실 압축, 멤버 hash, 공개 파일 hash, 로컬 캐시 목록 | 원본 파일을 복원·검증하는 공개 패키지 |

세 translator의 실제 학습 가중치·history·training inputs·train report는 [`../training/run_fit1000_val200`](../training/run_fit1000_val200/)에 원래 경로대로 보존한다.

## 기존 보고서 정정 및 해석상 주의

기존 `fit1000_eval160/REPORT.md`는 원본 보존을 위해 수정하지 않았다. 아래 정정을 함께 적용한다.

- 실제 optimizer는 `training.py`의 **Adam**이다. 기존 보고서의 AdamW 표기는 오류다. 현재 Affine은 identity 초기화, lr 0.001, 8 epochs 중 validation state MSE가 최소인 epoch 6이다.
- 이전 +1..+10 보고서와 최신 GT-visible +1..+5 비교는 평가 창과 가시성 조건이 다르다. 표의 숫자가 같아야 하는 것은 아니다.
- `requested_metrics_20261001`의 retention은 matched video 평균의 비율이다. 최종 요청인 **case별 비율의 평균**은 `paper_metrics_20261001/case_recovery.csv`와 `quality.csv`에 있다. 두 정의를 혼용하지 않는다.
- Base+-native/Full Replay는 비교 참조다. Small보다 낮은 영상도 있어 수학적·경험적 절대 상한으로 부르지 않는다.
- Warm score가 0인 비율은 undefined다. 매우 작은 양수일 때 회복률이 폭증할 수 있으며 epsilon·clipping으로 숨기지 않았다. 음수 격차 분모의 gap recovery도 의도한 단조적 해석이 불가능하다.
- 초기·후기 표는 GT 가시성에 따라 영상 수가 달라질 수 있다. 시간에 따른 악화를 주장하려면 같은 영상의 gap 변화와 CI를 확인한다.
- 160개 원본 평가 중 LVOS 한 case의 prompt가 0이 아니다. `native159_prompt0`은 이를 제외한다. 영상당 객체 하나의 독립 추적이며 공식 다객체 동시 평가가 아니다.
- MOSE는 local train/development이며 공식 validation 서버 결과가 아니다. 기존 F는 표준 DAVIS threshold이며 MOSEv2 수정 F와 구분한다. DAVIS train은 탐색적 평가이고 대부분 영상은 이전 평가 이력이 있다.
- 대규모 다운로드 bank는 22,644 scheduled pairs 중 19,525 pairs, 78,100 method rows를 완료하고 사용자 요청으로 중단했다. 초기 5프레임만 있고 native/source reference·장기 추론·공정한 비용 계측은 없다. 논문 최종 held-out 결과로 취급하지 않는다.
- 미실행 baseline, 공식 제출, Switch-B, 25/50/75% switch 기반 checkpoint 선택 등은 생성하지 않았다. 상세 승인 필요 목록은 paper `metrics.json`의 `requires_user_approval`에 보존한다.

## 원시 파일 복원·검증

`publication_20261002/raw_runs/*.tar.gz`의 `test10/...` 멤버는 저장소 루트에서 압축을 풀면 원래 상대 경로로 복원된다. `translator_source.tar.gz`와 `sam2_source.tar.gz`는 각각 `translator/`, `sam2/` 경로를 사용한다. 압축 파일 자체와 각 멤버의 SHA-256은 `published_files.csv`, `archive_members.csv`에 있다. 패키징 과정에서 모든 멤버를 다시 읽어 원본 바이트와 일치함을 검증했다.

GitHub에 올리지 않은 약 20GB·1,280개 예측 마스크 캐시는 사용자 승인에 따라 로컬에 보존했다. 경로·크기·기존 평가 시 기록된 hash는 `unpublished_prediction_caches.csv`에 있다. 원본 RGB/GT 데이터셋, 다운로드 학습 pair bank, 기반 SAM checkpoint, 가상환경도 결과 패키지와 분리된 입력이다.

프레임 CSV·manifest만으로 구간별 표와 paired CI는 다시 계산할 수 있다. 픽셀 수준 metric 재계산에는 원시 예측 캐시와 GT가 추가로 필요하다. 픽셀 재평가를 할 때 pickle payload는 신뢰할 수 있는 원본 및 hash를 확인한 경우에만 읽는다.

```bash
python test10/tools/compare_affine_windows.py \
  --paper-dir test10/results/paper_metrics_20261001 \
  --native-dir test10/results/fit1000_eval160 \
  --output-dir /tmp/test10-affine-recomputed
```

이 재집계는 Python·NumPy만 필요하며 GPU·모델·원본 RGB를 사용하지 않는다. 기존 output 디렉터리를 덮어쓰지 않는다.
