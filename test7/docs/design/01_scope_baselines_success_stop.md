# 01 연구 범위 동결: Baseline·성공 기준·중단 기준

> 상태: **FROZEN v1.1 — 2026-09-21**
> 적용 범위: SAM 2.1 Small → Base+, DAVIS 2017·MOSEv2·LVOS v2, nonlinear state translator  
> 목적: Project task 01의 연구 질문과 판정 규칙을 실험 결과를 보기 전에 고정한다.

## 1. 연구 질문과 범위

Small이 frame `t`까지 축적한 객체별 memory/state를 nonlinear translator로 Base+가 사용할 state로 바꾸면, Base+가 과거 RGB 전체를 replay하지 않고 `t+1`부터 추적을 이어갈 수 있는가?

핵심 주장은 tensor 복원 자체가 아니다. 다음 세 조건을 함께 본다.

1. Base+로 전환할 이유가 있는 어려운 상황이 실제로 존재한다.
2. 번역된 state가 Direct Copy와 강한 mask/prompt/replay 대안보다 후속 추적에 유용하다.
3. Full Replay보다 과거 재처리량·handoff 시간·전송 비용을 줄인다.

새 Linear/Ridge translator 개발, model router, backbone 재학습은 현재 제안 범위가 아니다. Linear/Ridge의 과거 결과는 nonlinear이 우수하다는 증거가 아니라 역사적 관찰로만 남긴다.

## 2. 모든 비교군에 공통인 공정성 규칙

- 동일한 `(video, registered object set, prompt timeline, switch frame, future frames)` manifest를 사용한다.
- Source와 Target이 같은 영상·frame 순서·preprocessing을 사용한다는 것은 실험 manifest가 보장한다. 영상 hash와 `num_frames`·높이·너비는 translator 입력이나 handoff payload가 아니라 dataset 준비 단계의 assertion으로만 취급한다.
- switch 이전에 등록된 **모든 객체**를 비교군 입력 범위에 포함한다. 최근 창에 보이지 않았다는 이유로 객체를 제외하지 않는다.
- 미래 GT mask는 handoff 입력 선택에 사용하지 않고 평가에만 쓴다.
- source 예측 mask를 쓰는 방법은 GT가 아니라 해당 실행에서 얻은 source prediction을 사용한다.
- 실제 point·box·mask prompt와 correction은 발생한 frame ID와 함께 보존한다.
- 전체 평균과 함께 visible, GT-absent, occlusion, reappearance, distractor, fast motion, long absence를 나눠 보고한다.
- 같은 영상의 여러 switch를 독립 영상처럼 세지 않고 video-clustered confidence interval을 사용한다.
- 각 방법의 전달 bytes, 재처리 frame 수, wall time, peak VRAM을 정확도와 함께 기록한다.
- 입력 정보가 서로 다른 방법을 같은 이름으로 부르지 않는다. 특히 `Last-Visible`과 `Original-Prompt(s)+Replay-k`를 구분한다.

## 3. 최종 Baseline

이 절의 **주 비교군**은 아래 표에 명시한 항목으로 한정한다. 이는 팀이 합의한
`Source-only → Target-native/Full Replay → Direct State Copy → 객체별 anchor
재인코딩 → Original+Replay-k → Nonlinear Translator` 구조다. `All Original
Prompts`, `Translation + Short Replay`, Target Reset은 주 비교군의 성능을 대신하는
항목이 아니라, interaction 또는 구현 실패를 해석하기 위한 선택적 분석으로 분리한다.

### 3.1 전환 필요성과 참조 성능

| 비교군 | Target 또는 실행이 받는 데이터 | 답하는 질문 | 해석 주의 |
|---|---|---|---|
| **Source-only** | Small의 기존 state를 그대로 계속 사용 | Base+로 바꿀 필요가 있는가? | 수학적 하한이 아니다. Small이 더 잘할 수 있다. |
| **Base+-native / Full Replay** | 과거 RGB 전체와 실제 prompt/correction timeline | Base+가 과거를 직접 봤을 때의 성능과 비용은? | 수학적 상한이 아니다. reference/oracle이라는 표현은 실행 조건을 뜻한다. |

Base+-native가 Source-only보다 유리하지 않은 slice에서는 Small→Base+ 전환의 필요성을 주장하지 않는다.

### 3.2 상태 번역 필요성

| 비교군 | 전달 데이터 | 답하는 질문 | 객체 정보 범위 |
|---|---|---|---|
| **Direct State Copy** | Small의 `maskmem_features`와 `obj_ptr`를 값 변경 없이 복사하고, 필수 discrete history/metadata는 보존. Target PE는 Base+ 정책으로 생성 | 학습된 번역이 정말 필요한가? | source memory bank에 등록된 모든 객체 |
| **Moment-Matched Copy** | 학습 split의 paired state에서 고정한 Source/Target별 평균·표준편차로 `maskmem_features`와 `obj_ptr`만 affine 보정. 나머지 payload와 Target PE 정책은 Direct와 동일 | 단순한 값 분포 차이만 보정해도 충분한가? | Direct와 완전히 동일 |
| **Nonlinear Translator** | Direct와 동일한 complete state 범위에서 `maskmem_features`와 `obj_ptr`를 component별 nonlinear mapper로 변환 | 제안 방식이 단순 분포 보정보다 실제 후속 성능을 개선하는가? | Direct와 완전히 동일 |

`object_score_logits`는 learned translator와 Moment-Matched Copy의 입력·출력이 아니며 Source 진단 기록으로만 보존한다. Source `pred_masks`도 Target history 또는 decoder refinement 입력으로 주입하지 않고 외부 표시 archive에만 보존한다. `maskmem_pos_enc`는 Target 정책으로 재생성하고, 새 mask·score·memory는 Target이 전환 후 frame 또는 correction replay에서 생성한다.

**Moment-Matched Copy 규칙.** 이 방법은 학습 파라미터가 없는 통계적 calibration baseline이다. spatial memory는 conditioning/non-conditioning별 **channel 단위**, object pointer는 conditioning/non-conditioning별 **dimension 단위**로, 유효 record만 사용해 학습 video의 paired state에서 `(μ_source, σ_source, μ_target, σ_target)`를 한 번 계산하고 고정한다. 각 test record에는 다음만 적용한다.

```text
x_target_like = ((x_source - μ_source) / max(σ_source, ε)) × σ_target + μ_target
```

test video의 Base+-native state, future frame, test-set 통계는 평균·표준편차 계산에 쓰지 않는다. 이 baseline은 handoff 시 과거 RGB replay를 수행하지 않으며, 통계 파일의 bytes와 runtime 변환 시간도 함께 기록한다.

### 3.3 객체별 anchor 재인코딩 대안

| 비교군 | Target에 주는 데이터 | 답하는 질문 | 부재·late-prompt 처리 |
|---|---|---|---|
| **Original-Prompt(s) Only** | 객체별 최초 실제 prompt와 해당 RGB/frame ID | 처음 지정한 정보만으로 충분한가? | frame 0으로 고정하지 않는다. late-prompt 객체도 자기 최초 prompt를 받는다. |
| **Last-Visible Source Mask** | 객체별 마지막 비어 있지 않은 source 예측 mask와 해당 RGB/frame ID | 각 객체의 최신 유효 관측 하나면 충분한가? | switch 직전 부재해도 그 객체의 마지막 관측을 찾는다. 없으면 Original로 fallback하고 횟수를 보고한다. |
| **Original + Last-Visible** | 위 두 anchor의 합집합 | 신뢰 가능한 최초 지정과 최신 관측의 조합이면 충분한가? | 중복 `(object, frame)`은 한 번만 처리한다. |

Original과 Last-Visible은 각각 따로 평가한다. 결합군이 단독군을 대신하면 어느 정보가 기여했는지 알 수 없기 때문이다.

### 3.4 제한된 replay 대안

| 비교군 | Target에 주는 데이터 | 답하는 질문 | 위치 |
|---|---|---|---|
| **Original-Prompt(s) + Replay-4** | 모든 객체의 original anchor와 switch 전 최근 RGB 4장, 창 안의 실제 correction | 원래 객체 지정에 짧은 최신 문맥을 더하면 충분한가? | 주요 실용 경쟁군 |
| **Original-Prompt(s) + Replay-8** | 모든 객체의 original anchor와 switch 전 최근 RGB 8장, 창 안의 실제 correction | SAM 2의 제한된 recent-memory horizon과 비슷한 길이의 문맥이 필요한가? | 주요 실용 경쟁군 |
| **Original-Prompt(s) + Replay-16** | 모든 객체의 original anchor와 switch 전 최근 RGB 16장, 창 안의 실제 correction | 더 긴 최근 문맥이 accuracy–cost trade-off를 계속 개선하는가? | 주요 실용 경쟁군 |

이 세 방법에서 `Replay-k`는 임의의 객체를 고르는 방식이 아니라 **시간 기준 최근 RGB
창을 target이 다시 처리하는 비용**이다. 모든 switch 이전 등록 객체는 객체별 original
anchor로 먼저 보장하고, 이후 최근 창을 시간순으로 재처리한다. 즉, 최근 창에 특정 객체가
없어도 그 객체의 등록 정보 자체는 빠지지 않는다.

`k={4,8,16}`은 짧은 문맥, 기본 SAM 2 recent-memory 범위와 비슷한 문맥, 그보다 긴
문맥을 각각 나타내도록 고정한다. `8`은 SAM 2 memory bank와 같은 계산을 재현한다는
뜻이 아니라, `num_maskmem=7` 주변의 최근 문맥 길이를 점검하기 위한 실용적인 중간점이다.
`k=1`/Last-Mask와 original anchor 없이 최근 창만 재처리하는 Recent-Window Replay-k는
입력 정보가 너무 약하고 객체 coverage를 보장하지 않으므로 **최종 비교군에서 제외**한다.
이 방법들은 이후에도 결과 표·평균·Pareto 순위에 넣지 않는다.

### 3.5 선택적 분석 (주 비교군이 아님)

- **All Original Prompts:** correction이 있는 영상에서, switch 이전의 모든 실제
  prompt/correction을 target에 재적용한다. Original-Prompt(s) Only의 의미를 바꾸지
  않기 위한 interaction-history 민감도 분석이다.
- **Translation + Short Replay:** 번역된 complete state에 최근 RGB `k`장을 더해
  source가 이미 버린 정보를 짧은 replay로 보완하는 hybrid다. translation-only가
  부족할 때의 후속 방법이며, 주 비교군을 통과한 뒤 Pareto 분석에서만 평가한다.

### 3.6 진단군과 정확성 검사

| 항목 | 역할 |
|---|---|
| **Target Reset / Empty-mask proxy** | 아무 기억 없이 바꾸는 경우의 실패 형태를 보는 진단군. 정식 reset으로 과장하지 않는다. |
| **Same-checkpoint export→inject** | Base+ state를 Base+에 주입했을 때 동일 continuation이 나오는지 확인하는 구현 정확성 검사. 경쟁 baseline이 아니다. |

## 4. 성공 판정

성공은 아래 단계로 나눠 판정한다. 한 단계의 성공을 다른 단계의 성공으로 바꾸어 쓰지 않는다.

### A. 구현 성공

- same-checkpoint self-injection이 정해진 tolerance 안에서 native continuation과 일치한다.
- 과거 RGB backbone replay가 0회이며, 모든 등록 객체와 유효 memory record가 누락 없이 복원된다. Prompt timeline은 handoff state가 아니라 interaction manifest가 보존한다.
- 다객체, late prompt, 부재·재등장, prompt correction에서도 fail-closed 계약을 지킨다.

이 단계는 translator 성능 성공이 아니라 task 06의 구현 gate다.

### B. 전환 필요성 성립

- 사전 정의한 hard-event slice 중 적어도 하나에서 Base+-native가 Source-only보다 낫다.
- 차이는 영상 단위로 군집화한 95% confidence interval과 effect size를 함께 보고한다.
- Base+-native 이득이 없는 slice에서는 translator 성공을 주장하지 않는다.

### C. Nonlinear translator 유효성

held-out video에서 다음을 모두 확인한다.

1. Direct State Copy보다 downstream J&F 또는 사건별 continuity 지표가 개선된다.
2. tensor loss 개선만이 아니라 switch 후 1/5/20 frame, 재등장 recovery, GT-absent false positive 중 관련 지표가 함께 개선된다.
3. 가장 강한 non-full-replay 대안과 accuracy–cost Pareto 비교에서 지배되지 않는다.
4. Full Replay보다 과거 재처리 frame 수가 적고, prefix 길이에 따른 handoff latency crossover를 보고한다.
5. DAVIS만의 단일 성공으로 일반화를 주장하지 않는다. MOSEv2 또는 LVOS v2의 long-term/hard condition에서 방향이 재현되어야 주 방법의 일반화 근거로 사용한다.

주요 차이는 video-clustered 95% CI와 함께 보고한다. 동률 주장은 사전에 정한 `1.0 J&F point` 비열등 margin 안에서만 사용하고, 그 경우 비용 이득을 반드시 함께 제시한다.

## 5. 중단·수정 기준

### 5.1 즉시 학습을 멈추고 구현을 수정하는 조건

- same-checkpoint injection이 일치하지 않거나 객체·frame 정렬 오류가 난다.
- 미래 GT, test video 또는 Base+-native future state가 translator 입력·선택에 누출된다.
- checkpoint/config/preprocessing/prompt timeline 또는 dataset/pair manifest가 양쪽에서 다르다.
- evaluator가 공식 mask 규칙과 일치하지 않거나 실행 재현 정보가 없다.

이 조건은 연구 아이디어의 실패가 아니라 실험 무결성 실패다. 원인을 고치고 해당 결과를 폐기한 뒤 다시 실행한다.

### 5.2 Small→Base+ 전환 주장을 중단하는 조건

- hard-event slice에서도 Base+-native가 Source-only보다 안정적인 이득을 보이지 않는다.
- Base+의 추가 계산 비용을 감수할 실제 성능 이점이 없다.

이 경우 translator 학습을 계속해도 “왜 전환하는가”에 답할 수 없으므로 이 모델 쌍의 주장을 축소하거나 다른 pair/scenario로 전환한다.

### 5.3 Learned translator 주장을 중단하거나 전환하는 조건

- Direct State Copy가 반복 평가에서 Base+-native와 사실상 같으면 nonlinear translator가 필요하지 않다. 결과는 direct interoperability protocol로 보고한다.
- nonlinear이 state MSE만 줄이고 downstream mask·continuity를 개선하지 못하면 loss/component 설계를 수정한다.
- Original+Last-Visible 또는 Original+Replay-k가 같은 정확도에서 더 싸고 안정적이면 nonlinear의 실용 우위를 주장하지 않는다.
- 필요한 short replay 길이가 Full Replay에 가까워져 latency·재처리 이득이 사라지면 translation-only 주장을 중단한다.
- 한 영상 또는 한 객체에만 성립하고 held-out/long-term dataset에서 재현되지 않으면 일반화 주장을 중단한다.

번역 단독은 부족하지만 짧은 replay에서 분명한 Pareto 이득이 있으면 실패로 숨기지 않고 **Translation + Short Replay hybrid**로 연구 결론을 수정한다.

## 6. Task 01 완료 선언

다음 항목을 본 문서로 고정했다.

- [x] 모델 방향, 데이터셋, nonlinear 연구 범위
- [x] 경쟁 baseline·진단군·정확성 검사 구분
- [x] 모든 등록 객체를 보장하는 입력 규칙
- [x] 미래 GT 금지와 동일 manifest 원칙
- [x] downstream·continuity·cost 기반 성공 기준
- [x] 구현 중단, 모델 전환 no-go, learned translator no-go 기준

실험 결과에 따라 결론은 바뀔 수 있지만, 결과를 본 뒤 유리하게 판정 규칙을 바꾸지는 않는다. 변경이 필요하면 날짜·이유·영향받는 run을 decision log에 남긴다.
