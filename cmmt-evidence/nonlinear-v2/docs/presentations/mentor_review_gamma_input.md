# CMMT 멘토 검토용 Gamma 입력 패키지

## 사용 방법

1. Gamma에서 `프레젠테이션`을 선택한다.
2. `텍스트로 붙여넣기`의 `기본` 모드를 유지한다.
3. 아래 `Gamma에 그대로 붙여 넣을 내용`만 복사해 입력창에 넣는다.
4. 10장으로 생성하고 이미지 설정은 `이미지 자리표시자` 또는 `이미지 없음`을 선택한다.
5. 생성 후 `자료 배치표`에 따라 실제 연구 자료와 링크를 넣는다.

## Gamma에 그대로 붙여 넣을 내용

다음 내용을 한국어 학술 연구 발표용 10장 프레젠테이션으로 구성하세요.

청중은 이 연구를 처음 접하는 인공지능 연구 멘토입니다. 목적은 연구 문제, 현재까지 완성한 실험 기반, 아직 하지 않은 일, 다음 실험 순서를 설명하고 연구 방향을 검토받는 것입니다.

각 슬라이드는 아래 번호와 순서를 그대로 유지하세요. 한 슬라이드에는 하나의 핵심 메시지만 넣고 긴 문단은 만들지 마세요. 제목과 핵심 문장은 크게 표시하고, 세부 설명은 최대 네 줄로 제한하세요. Small은 파란색, Base+는 주황색, Translator는 보라색으로 일관되게 표시하세요. 완료 단계는 짙은 색, 이후 단계는 회색으로 표현하세요. 임의의 AI 사진이나 장식 이미지는 만들지 말고 연구 흐름도, 비교도, 결과표를 사용하세요. 아래의 `시각 구성` 문장은 슬라이드 본문이 아니라 배치 지시입니다.

### 슬라이드 1. Cross-Model Memory Translator

SAM 2 Small에서 Base+로 비디오 메모리 전달

김규도  |  멘토 검토용 연구 진행 보고  |  2026-09-23

시각 구성: 여백이 넓은 표지. Small, Translator, Base+를 나타내는 작은 색상 표식만 사용한다.

### 슬라이드 2. 비디오 중간 모델 교체의 문제

비디오 중간에 모델을 바꾸면 새 모델은 이전 물체의 기억을 가지고 있지 않다.

Small이 과거 프레임과 지정된 객체를 처리한다.

중간 시점에 Base+로 교체하면 Base+에는 누적된 temporal memory가 없다.

가장 확실한 복구 방법은 과거 RGB 전체를 Base+로 다시 처리하는 것이지만, 긴 영상에서는 지연과 연산량이 커진다.

시각 구성: 긴 비디오 타임라인 중앙에 switch 지점을 두고, 왼쪽 Small과 오른쪽 Base+를 배치한다. Base+ 쪽 memory는 비어 있는 모습으로 표현한다.

### 슬라이드 3. CMMT의 제안

Source memory를 Target이 사용할 수 있는 memory로 번역해 다음 프레임부터 추론을 이어간다.

기존 방법: 과거 RGB 전체를 Base+로 다시 처리한다.

CMMT: Source state b_t를 번역해 Target state 추정값 a_hat_t = T(b_t)를 만든다.

목표는 tensor 오차만 줄이는 것이 아니라 후속 J&F와 객체 정체성, 가림 이후 회복을 유지하면서 replay 비용을 줄이는 것이다.

시각 구성: 왼쪽에는 Full Replay, 오른쪽에는 CMMT를 배치한 비교도. CMMT 경로는 Small memory, Translator, Base+ continuation 순서로 표시한다.

### 슬라이드 4. SAM 2에서 전달할 상태

현재 계약은 spatial memory와 object pointer를 번역 대상으로 사용한다.

번역 대상: maskmem_features, obj_ptr

조립 metadata: frame index, object ID, slot order, conditioning 여부, validity, switch frame

Base+가 생성: maskmem_pos_enc와 전환 이후의 새 mask, score, memory

전달하지 않음: 과거 RGB 전체, 영상 크기, fingerprint, 과거 raw mask와 진단 score

시각 구성: Small history에서 두 state가 CanonicalState와 Nonlinear Translator를 거쳐 Base+ history로 들어가는 단순 흐름도.

### 슬라이드 5. 현재 연구 범위

현재 버전은 SAM 2 Small에서 Base+로의 Nonlinear memory translation을 검증한다.

Source: SAM 2.1 Small

Target: SAM 2.1 Base+

학습·in-domain: MOSEv2, LVOS v2

external: VOST(primary)

제안 방식: Nonlinear Translator

초기 Tiny→Large, Ridge, Residual MLP 결과는 역사적 파일럿이며 현재 버전의 성능 결과에 포함하지 않는다.

시각 구성: Source, Target, Method, Dataset 네 항목을 평평한 2열 구성으로 정리한다. 대시보드 형태의 작은 카드 격자는 사용하지 않는다.

### 슬라이드 6. 연구 단계와 현재 위치

기존 runtime 검증과 VOST core gate를 마쳤고, PUMaVOS·M³-VOS complementary external contract를 검증한 뒤 paired-state를 수집한다.

01 연구 범위와 성공 기준: 완료

02 State I/O 계약: 완료

03 Benchmark v1.0: 완료 / v1.1 VOST onboarding: 진행 중

06 Runtime handoff: 완료

07 MOSE/LVOS paired-state 수집: 다음 단계

08 공통 baseline 평가

09 Nonlinear Translator 학습

10 전체 데이터셋 평가와 논문 작성

시각 구성: 왼쪽에서 오른쪽으로 이어지는 단계 흐름. 완료 단계는 짙은 색, Task 07은 강조색, 이후 단계는 회색으로 표현한다.

### 슬라이드 7. 완료한 실험 기반

학습 결과를 비교하기 전에 범위, 상태 계약, 데이터 조건을 고정했다.

Task 01: 연구 질문, baseline, 성공 기준, 중단 기준을 확정했다.

Task 02: maskmem_features와 obj_ptr의 export, 검증, Target 조립 규칙을 확정했다.

Task 03 v1.0: MOSEv2·LVOS v2 manifest, video-level split, checksum을 고정했다.

Task 03: MOSE/LVOS fit·dev와 sealed final, VOST primary external 역할을 고정했고 VOST loader/evaluator contract 검증까지 완료했다. PUMaVOS·M³-VOS external onboarding 전까지 Task 03은 진행 중이다.

전체 RGB, prompt mask, object ID, switch frame loader 검증 결과는 failure_count=0이다.

시각 구성: Task 01, 02, 03을 위에서 아래로 연결하고 마지막에 검증 완료 표시를 둔다.

### 슬라이드 8. Runtime handoff 검증

상태를 꺼내고 다시 넣는 runtime이 후속 예측을 훼손하지 않는다는 것을 확인했다.

실행 흐름: Source export, CanonicalState, Target 조립과 주입, 후속 frame 추론

Base+ self-injection 결과: MSE 0, 최대 오차 0, binary IoU 1.0

다객체와 late prompt: 후속 48 frames exact

객체 부재와 재등장: 후속 45 frames exact

전환 이후 correction: 후속 52 frames exact

전환 이전 correction: replay 이후 후속 51 frames exact

반복 switch: 첫 전환 뒤 61 frames, 두 번째 전환 뒤 51 frames exact

상태 주입 중 과거 backbone 호출: 0

시각 구성: 상단에는 네 단계 runtime 흐름, 하단에는 결과표를 배치한다. exact는 native Base+와 후속 logit이 정확히 일치했다는 뜻으로 각주에 설명한다.

### 슬라이드 9. Direct Copy가 보여준 번역 필요성

Small state는 Base+ history에 구조적으로 들어갔지만 후속 객체 추적에는 실패했다.

조건: 과거 단일 video direct-copy pilot, switch 뒤 61 frames

spatial-memory cosine: 0.0211

object-pointer cosine: -0.0220

Base+-native 대비 후속 binary IoU: 0.0

해석: tensor shape가 같아도 Small과 Base+의 state 의미는 호환되지 않는다.

이 결과는 현재 범위에서 제외한 과거 단일-case Direct Copy 파일럿이며 Nonlinear Translator의 성공 결과가 아니다. 발표용 성능 근거로 사용하지 않는다.

시각 구성: 왼쪽의 Small state가 Base+에 주입되는 흐름은 성공 표시, 오른쪽의 후속 mask continuation은 실패 표시. 결과 숫자 세 개를 크게 보여준다.

### 슬라이드 10. 다음 실험과 멘토 검토 항목

다음 단계에서는 동일한 protocol로 paired state와 baseline을 준비한 뒤 Nonlinear Translator를 평가한다.

Task 03: PUMaVOS download·manifest·loader·metric 계약 검증

Task 07: MOSEv2/LVOS v2 fit/dev에서 Small과 Base+ paired-state 수집

Task 08: 모든 baseline을 같은 evaluator와 switch case에서 평가

Task 09: Nonlinear 후보 학습, validation, 구조 선정

이후: MOSE/LVOS sealed in-domain, VOST primary·PUMaVOS/M³-VOS complementary external 평가, ablation

멘토 검토 질문

1. maskmem_features와 obj_ptr를 번역 대상으로 삼는 것이 충분한가?
2. 현재 baseline 구성이 공정한가?
3. 첫 Nonlinear 후보를 MLP로 시작하는 순서가 적절한가?
4. 전체 평가 전에 추가할 실패 조건이나 ablation이 있는가?

시각 구성: 왼쪽에는 Task 07부터 전체 평가까지의 순서, 오른쪽에는 멘토 검토 질문을 배치한다.

## 생성 후 자료 배치표

| 슬라이드 | 사용할 자료 | 처리 방법 |
|---|---|---|
| 2 | CMMT 문제 타임라인 | Gamma native diagram으로 작성. 외부 이미지는 사용하지 않는다. |
| 3 | Full Replay와 CMMT 비교 | Gamma native diagram으로 작성. 수식은 `a_hat_t = T(b_t)` 하나만 사용한다. |
| 4 | State Assembly Map | 아래 원문을 참고해 발표용 단순 흐름도로 다시 그린다. 원문 전체 화면을 축소해서 넣지 않는다. |
| 6 | Project Board 단계 | Gamma native process diagram으로 작성한다. |
| 7 | Task 03 freeze 결과 | 아래 Task 03 보고서에서 데이터 수와 `failure_count=0`만 사용한다. |
| 8 | Task 06 결과표 | 아래 Task 06 보고서의 표를 6행 이내로 옮긴다. |
| 9 | Direct Copy 실패 | 검증된 예측 PNG가 현재 버전에 없으므로 임의 mask 이미지를 만들지 않는다. 주입 성공과 continuation 실패를 나타내는 native diagram과 세 수치만 사용한다. |
| 10 | 검토 질문 | 위 네 질문을 그대로 사용한다. |

## 검증 원문과 링크

- State Assembly Map: https://memorybridge-team.github.io/vos-memory-translator-nonlinear/architecture/cmmt-state-assembly-map.html
- Task 03 현재 보고서: https://github.com/memorybridge-team/vos-memory-translator-nonlinear/tree/main/reports/tasks/03_benchmark
- Task 03 Benchmark v1.1 addendum: https://github.com/memorybridge-team/vos-memory-translator-nonlinear/blob/main/reports/tasks/03_benchmark/milestones/2026-09-24_v1_1_addendum.md
- Task 06 최종 통합본: https://github.com/memorybridge-team/vos-memory-translator-nonlinear/blob/main/reports/tasks/06_runtime/FINAL_REPORT.md
- Task 06 edge cases·Direct Copy: https://github.com/memorybridge-team/vos-memory-translator-nonlinear/tree/main/reports/tasks/06_runtime/runs/2026-09-21_edge_case_and_direct_injection
- Task 06 merged PR: https://github.com/memorybridge-team/vos-memory-translator-nonlinear/pull/8
- 결과 홈페이지: https://memorybridge-team.github.io/vos-memory-translator-nonlinear/

## 로컬 원문

- `docs/architecture/cmmt-state-assembly-map.html`
- `reports/tasks/03_benchmark/milestones/2026-09-23_benchmark_v1_0_freeze.md`
- `reports/tasks/03_benchmark/milestones/2026-09-24_v1_1_addendum.md`
- `reports/tasks/06_runtime/FINAL_REPORT.md`
- `reports/tasks/06_runtime/runs/2026-09-21_edge_case_and_direct_injection/README.md`

## 생성 후 사실 검수 체크리스트

- 현재 연구 범위를 Small→Base+로 표현했는가?
- 초기 Tiny→Large/Ridge/Residual-MLP를 현재 결과로 표시하지 않았는가?
- Nonlinear Translator가 아직 학습 전이라고 명확히 표시했는가?
- self-injection 결과와 cross-model Direct Copy 결과를 구분했는가?
- Direct Copy 한 사례의 결과를 전체 데이터셋 결론으로 확대하지 않았는가?
- `exact`가 native Base+와의 일치이지 ground-truth 정답을 뜻하지 않는다고 설명했는가?
- Task 03 v1.1 VOST gate와 이후 Task 07 순서가 구분되어 있는가?
