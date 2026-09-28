# CMMT 멘토 검토용 PowerPoint Brief

## 1. 발표 목적

이 발표는 연구를 처음 접하는 인공지능 연구 멘토에게 다음을 설명하고 검토받기 위한 자료다.

- CMMT가 해결하려는 문제와 제안 아이디어
- 현재 Small→Base+ 버전에서 완료한 실험 기반
- Runtime handoff가 정확하게 작동한다는 증거
- 현재까지 확인한 Direct Copy의 한계
- Task 07 이후 실험 순서와 연구 설계

이 발표는 Nonlinear Translator의 성공 결과 발표가 아니다. 현재는 번역기 학습 전이며, 실험 기반과 Runtime 검증을 마친 단계다.

## 2. 현재 연구 범위

- Source: SAM 2.1 Small
- Target: SAM 2.1 Base+
- 제안 방식: Nonlinear Translator
- 학습·in-domain: MOSEv2, LVOS v2
- external: VOST(primary)
- 완료 Task: 01 Scope, 02 State I/O, 06 Runtime
- 현재 Task: 03 PUMaVOS external contract 검증; 이후 07 Paired-state collection

초기 Tiny→Large, Ridge, Residual MLP와 rare-event 결과는 역사적 파일럿이다. 현재 버전의 성능 결과로 사용하지 않는다.

## 3. 시각 디자인

- 화면비: 16:9
- 배경: 흰색 또는 아주 밝은 회색
- Small: 파란색
- Base+: 주황색
- Translator: 보라색
- 완료 단계: 짙은 남색 또는 녹색
- 현재 단계 Task 03 v1.1: 강조색; 다음 Task 07은 별도 표시
- 미래 단계: 회색
- 표지 제목은 크게, 나머지 요소는 최소화
- 한 슬라이드에는 핵심 메시지 하나만 배치
- 장식용 사진, 일반적인 AI 이미지, 데이터센터 이미지를 사용하지 않음
- 흐름도와 결과표는 수정 가능한 PowerPoint 기본 도형과 표로 제작

## 4. 슬라이드 명세

### 슬라이드 1. Cross-Model Memory Translator

제목: Cross-Model Memory Translator

부제: SAM 2 Small에서 Base+로 비디오 메모리 전달

하단: 김규도 | 멘토 검토용 연구 진행 보고 | 2026-09-23

구성: 여백이 넓은 표지. Small, Translator, Base+를 나타내는 작은 색상 요소만 사용한다.

### 슬라이드 2. 비디오 중간 모델 교체의 문제

핵심 문장: 비디오 중간에 모델을 바꾸면 새 모델은 이전 물체의 기억을 가지고 있지 않다.

내용:
- Small이 과거 프레임과 사용자가 지정한 객체를 처리한다.
- 중간 시점에 Base+로 교체하면 Base+에는 누적된 temporal memory가 없다.
- 과거 RGB 전체를 Base+로 다시 처리하면 긴 영상에서 지연과 연산량이 증가한다.

구성: 긴 비디오 타임라인 중앙에 switch 지점을 두고 왼쪽 Small, 오른쪽 Base+를 배치한다. Base+의 memory가 비어 있는 상태를 표시한다.

### 슬라이드 3. CMMT의 제안

핵심 문장: Source memory를 Target이 사용할 수 있는 memory로 번역해 다음 프레임부터 추론을 이어간다.

내용:
- Full Replay: 과거 RGB 전체를 Base+가 다시 처리한다.
- CMMT: Source state `b_t`를 번역해 Target state 추정값 `a_hat_t = T(b_t)`를 만든다.
- 목표는 tensor 오차만 줄이는 것이 아니라 후속 J&F, 객체 정체성, 가림 이후 회복을 유지하면서 replay 비용을 줄이는 것이다.

구성: Full Replay와 CMMT를 나란히 비교한다.

### 슬라이드 4. SAM 2에서 전달할 상태

핵심 문장: 현재 계약은 spatial memory와 object pointer를 번역 대상으로 사용한다.

내용:
- 번역 대상: `maskmem_features`, `obj_ptr`
- 조립 metadata: frame index, object ID, slot order, conditioning 여부, validity, switch frame
- Base+가 생성: `maskmem_pos_enc`, 전환 이후의 새 mask, score, memory
- 전달하지 않음: 과거 RGB 전체, 영상 크기, fingerprint, 과거 raw mask, 진단 score

구성: Small history에서 두 state가 CanonicalState와 Nonlinear Translator를 거쳐 Base+ history로 들어가는 단순 흐름도. 첨부한 State Assembly Map 전체를 축소해서 넣지 않는다.

### 슬라이드 5. 현재 연구 범위

핵심 문장: 현재 버전은 SAM 2 Small에서 Base+로의 Nonlinear memory translation을 검증한다.

내용:
- Source: SAM 2.1 Small
- Target: SAM 2.1 Base+
- 방법: Nonlinear Translator
- 학습·in-domain: MOSEv2, LVOS v2
- external: VOST(primary)
- 초기 Tiny→Large, Ridge, Residual MLP는 현재 성능 결과에 포함하지 않는다.

### 슬라이드 6. 연구 단계와 현재 위치

핵심 문장: 기존 Runtime과 VOST gate를 검증했고, PUMaVOS·M³-VOS complementary external contract를 검증한 뒤 paired-state를 수집한다.

단계:
- 01 연구 범위와 성공 기준: 완료
- 02 State I/O 계약: 완료
- 03 Benchmark v1.0: 완료 / v1.1 VOST onboarding: 진행 중
- 06 Runtime handoff: 완료
- 07 MOSE/LVOS paired-state 수집: 다음 단계
- 08 공통 baseline 평가
- 09 Nonlinear Translator 학습
- 10 전체 데이터셋 평가와 논문 작성

구성: 왼쪽에서 오른쪽으로 이어지는 단계 흐름. Task 07을 강조한다.

### 슬라이드 7. 완료한 실험 기반

핵심 문장: 학습 결과를 비교하기 전에 범위, 상태 계약, 데이터 조건을 고정했다.

내용:
- Task 01: 연구 질문, baseline, 성공 기준, 중단 기준 확정
- Task 02: `maskmem_features`와 `obj_ptr`의 export, validator, Target 조립 규칙 확정
- Task 03 v1.0: MOSEv2·LVOS v2 manifest, video-level split, checksum 확정
- Task 03: fit/dev·sealed final·VOST primary external 역할 고정; VOST loader/evaluator contract 검증 완료. PUMaVOS·M³-VOS external onboarding 전까지 진행 중
- RGB, prompt mask, object ID, switch frame 전수 loader 검증 `failure_count=0`

구성: Task 01, 02, 03을 순서대로 연결하고 마지막에 검증 완료 결과를 표시한다.

### 슬라이드 8. Runtime handoff 검증

핵심 문장: 상태를 꺼내고 다시 넣는 Runtime이 후속 예측을 훼손하지 않는다는 것을 확인했다.

실행 흐름:
- Source export
- CanonicalState
- Target 조립과 주입
- 후속 frame 추론

결과:
- Base+ self-injection: MSE 0, 최대 오차 0, binary IoU 1.0
- 다객체와 late prompt: 후속 48 frames exact
- 객체 부재와 재등장: 후속 45 frames exact
- 전환 이후 correction: 후속 52 frames exact
- 전환 이전 correction: replay 후 후속 51 frames exact
- repeated switch: 첫 전환 뒤 61 frames, 두 번째 전환 뒤 51 frames exact
- state injection 중 과거 backbone 호출: 0

각주: `exact`는 native Base+와 후속 logit이 일치했다는 뜻이며 ground-truth 정답을 뜻하지 않는다.

구성: 상단에는 네 단계 흐름, 하단에는 간결한 결과표를 배치한다.

### 슬라이드 9. Direct Copy가 보여준 번역 필요성

핵심 문장: Small state는 Base+ history에 구조적으로 들어갔지만 후속 객체 추적에는 실패했다.

조건:
- 과거 단일 video pilot
- 한 객체
- switch frame 10
- 후속 61 frames

결과:
- spatial-memory cosine: 0.0211
- object-pointer cosine: -0.0220
- Base+-native 대비 후속 binary IoU: 0.0

해석:
- tensor shape가 같아도 Small과 Base+의 state 의미는 호환되지 않았다.
- 현재 범위에서 제외한 과거 Direct Copy pilot이며 전체 데이터셋 결론이나 발표용 성능 근거가 아니다.
- Nonlinear Translator의 성공 결과가 아니다.

구성: 왼쪽에는 state 주입 성공, 오른쪽에는 후속 mask continuation 실패를 표시한다. 검증된 현재 버전의 prediction PNG가 없으므로 임의 mask 이미지를 만들지 않는다.

### 슬라이드 10. 다음 실험과 멘토 검토 항목

핵심 문장: 동일한 protocol로 paired state와 baseline을 준비한 뒤 Nonlinear Translator를 평가한다.

다음 실험:
- Task 03: PUMaVOS download·manifest·loader·metric 계약 검증
- Task 07: MOSEv2/LVOS v2 fit/dev에서 Small과 Base+ paired-state 수집
- Task 08: 모든 baseline을 같은 evaluator와 switch case에서 평가
- Task 09: Nonlinear 후보 학습, validation, 구조 선정
- 이후: MOSE/LVOS sealed in-domain, VOST primary·PUMaVOS/M³-VOS complementary external 평가, ablation

멘토 검토 질문:
1. `maskmem_features`와 `obj_ptr`를 번역 대상으로 삼는 것이 충분한가?
2. 현재 baseline 구성이 공정한가?
3. 첫 Nonlinear 후보를 MLP로 시작하는 순서가 적절한가?
4. 전체 평가 전에 추가할 실패 조건이나 ablation이 있는가?

구성: 왼쪽에는 향후 단계, 오른쪽에는 검토 질문을 배치한다.

## 5. 사실 검수 기준

- 현재 범위는 Small→Base+다.
- 현재 버전에서 학습된 Nonlinear Translator 결과는 없다.
- Base+→Base+ self-injection은 Runtime 정확성 검증이다.
- Small→Base+ Direct Copy는 한 사례의 실패 pilot이다.
- `MSE 0`, `IoU 1.0`, `exact`는 native Base+와의 일치를 의미한다.
- 현재는 Task 03 v1.1 VOST gate이며, 완료 뒤 Task 07로 진행한다.
- 근거 없는 성능 그래프와 예측 mask를 생성하지 않는다.

## 6. 근거 문서

- State Assembly Map: `docs/architecture/cmmt-state-assembly-map.html`
- Task 03 freeze: `reports/tasks/03_benchmark/milestones/2026-09-23_benchmark_v1_0_freeze.md`
- Task 03 v1.1 addendum: `reports/tasks/03_benchmark/milestones/2026-09-24_v1_1_addendum.md`
- Task 06 final report: `reports/tasks/06_runtime/FINAL_REPORT.md`
- Task 06 edge cases and Direct Copy: `reports/tasks/06_runtime/runs/2026-09-21_edge_case_and_direct_injection/README.md`

공개 원문:

- https://memorybridge-team.github.io/vos-memory-translator-nonlinear/architecture/cmmt-state-assembly-map.html
- https://github.com/memorybridge-team/vos-memory-translator-nonlinear/blob/main/reports/tasks/03_benchmark/milestones/2026-09-23_benchmark_v1_0_freeze.md
- https://github.com/memorybridge-team/vos-memory-translator-nonlinear/blob/main/reports/tasks/03_benchmark/milestones/2026-09-24_v1_1_addendum.md
- https://github.com/memorybridge-team/vos-memory-translator-nonlinear/blob/main/reports/tasks/06_runtime/FINAL_REPORT.md
- https://github.com/memorybridge-team/vos-memory-translator-nonlinear/tree/main/reports/tasks/06_runtime/runs/2026-09-21_edge_case_and_direct_injection
