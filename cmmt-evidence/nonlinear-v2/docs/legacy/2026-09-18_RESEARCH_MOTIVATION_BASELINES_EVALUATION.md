# CMMT: 연구 필요성, 비교군, 객체·전환 조건별 평가

작성·근거 확인: 2026-09-18 KST.

후속 모델 쌍 결정: 같은 날 사용자의 재검토 요청으로 개발 순서를 **Tiny→Base+ 우선,
Tiny→Large 확장 검증**으로 변경했다. 아래 Tiny→Large 중심 Introduction은 당시 초안이며,
현재 pair별 역할·진행 조건은 [최신 실험 계획](../experimental_plan.md)의 모델 쌍 결정을 따른다.

이 문서는 SAM 2.1 Tiny→Large nonlinear state handoff의 연구 설계다. **코드로 확인한 사실, 문헌의 결과, 제안한 실험을 구분한다.** 새 비교군의 구현·성능 검증을 완료했다는 보고가 아니다.

## 1. 10번 프레임에서 등록한 객체는 이후 어떻게 저장되는가?

### 1.1 직접적인 답

**공식 predictor로 10~50번을 모두 추론했다면, 기본 설정에서는 해당 객체의 프레임별 예측 결과가 각각 남는다. 객체가 안 보인다고 그 객체의 기록 전체가 삭제되는 것은 아니다.**

예를 들어 다음과 같은 상황을 가정한다. 아래 프레임 구간은 설명용이며 실제 실험 결과가 아니다.

| 프레임 | 상황 | 해당 객체에 저장되는 결과 |
|---|---|---|
| 10 | 사용자가 객체 A를 처음 지정 | prompt 입력, conditioning output, memory/pointer 등 |
| 11~29 | 모델이 A가 보인다고 판단 | 각 프레임의 예측 mask logits, spatial memory, pointer, presence score |
| 30~40 | 모델이 A가 안 보인다고 판단 | 빈 binary mask에 해당하는 logits, 부재 점수, no-object 처리가 반영된 memory/pointer |
| 41~50 | 모델이 A가 다시 보인다고 판단 | 각 프레임의 새 예측과 memory/pointer |

추가 prompt가 없다면 10번은 `cond_frame_outputs[10]`, 11~50번은 `non_cond_frame_outputs[frame_idx]`에 저장된다. 각 프레임의 출력이 별도 항목이므로 마지막 마스크 하나로 계속 덮어쓰는 방식이 아니다. 단, 같은 프레임을 수정·재추론하면 그 프레임 항목은 갱신될 수 있다.

근거: 프로젝트가 고정한 공식 SAM 2 commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`의 [`propagate_in_video`](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L583).

### 1.2 “빈 마스크를 저장한다”의 정확한 뜻

SAM 2.1 Tiny/Large 기본 설정의 일반적인 자동 추론 경로에서는 다음과 같이 동작한다.

1. 모델의 `object_score_logits`가 0 이하이면 객체가 안 보인다고 판단한다.
2. 예측 mask logits에 큰 음수인 `NO_OBJ_SCORE=-1024`를 적용한다. `logit > 0`으로 화면에 표시하면 모든 픽셀이 배경인 빈 마스크다.
3. object pointer에는 학습된 `no_obj_ptr`가 사용된다.
4. memory encoder는 여전히 현재 영상 feature와 mask 신호를 처리한다. 설정에 따라 학습된 no-object spatial embedding도 반영된다.

따라서 **빈 마스크 = memory tensor 없음 또는 memory tensor 전체가 0**은 아니다. 또한 이것은 모델의 판단이다. 실제로는 객체가 보이는데 모델이 놓친 경우에도 빈 예측이 나올 수 있다.

`pred_masks`는 저장된 저해상도 예측 logits이며, 원본 해상도의 GT 마스크나 원본 RGB 자체가 아니다. 입력 prompt와 모델 예측도 서로 다른 데이터다. 근거: [`sam2_base.py`](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L358), [`compact output 저장`](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L779), [SAM 2.1 Tiny 설정](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/configs/sam2.1/sam2.1_hiera_t.yaml).

### 1.3 저장한 기록과 현재 읽는 기억은 다르다

프레임별 결과를 보관하는 것은 **실행 중 predictor의 기록 저장**이지 모델 가중치를 다시 학습하는 것이 아니다. 또 저장한 모든 프레임을 매번 attention에 넣지도 않는다.

- 기본 설정에서 오래된 conditioning output은 현재 memory read의 후보로 남는다.
- non-conditioning spatial memory는 최근 프레임 위주로 선택된다. `num_maskmem=7`은 전체 history를 일곱 프레임만 저장한다는 뜻이 아니다.
- object pointer는 spatial memory와 별도의 선택 규칙을 갖는다.
- prompt/객체 삭제, reset, 명시적인 메모리 정리 또는 커스텀 streaming 구현은 보존 동작을 바꿀 수 있다.

51번을 예측할 때는 오래된 10번 conditioning과 최근 기록 등이 사용될 수 있다. 20번 결과가 history에 남아 있다는 이유만으로 반드시 직접 읽히는 것은 아니다. 근거: [`_prepare_memory_conditioned_features`](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L525).

이 구분은 translator에도 적용된다. 모든 **등록 객체의 ID와 필요한 상태**를 보존하는 것이 목표이지, 모든 과거 frame tensor를 무조건 전송해야 한다는 뜻은 아니다. 앞으로만 추론하는 계약과 과거 프레임 수정까지 지원하는 interactive 계약은 필요한 보존 범위가 다르므로 명시적으로 구분한다.

## 2. Introduction에 사용할 연구 필요성

### 2.1 논문용 초안

Video Object Segmentation(VOS)은 지정된 객체의 마스크를 연속 프레임에서 예측하는 문제이며, 현재 영상뿐 아니라 과거 관측에서 형성한 시간적 정보를 활용한다. XMem은 장기 영상 처리를 위해 서로 다른 시간 규모의 메모리를 구성하고, Cutie는 객체 수준의 정보를 활용해 유사한 물체 사이의 혼동을 줄인다. SAM 2 역시 streaming memory를 통해 prompt로 지정된 객체의 정보를 이후 프레임으로 전달한다. 이들 연구에서 메모리는 단순한 출력 캐시가 아니라 후속 추론을 구성하는 상태라는 점이 중요하다. [XMem, ECCV 2022](https://arxiv.org/abs/2207.07115), [Cutie, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Putting_the_Object_Back_into_Video_Object_Segmentation_CVPR_2024_paper.html), [SAM 2](https://arxiv.org/abs/2408.00714).

실제 배포에서는 추론 정확도와 연산·통신 제약을 함께 고려해야 한다. EdgeTAM은 SAM 2 계열 모델의 on-device 실행에서 memory attention 비용을 줄이는 문제를 다루며, ECSeg은 네트워크와 장면 조건에 따라 edge와 cloud의 서로 다른 segmentation 모델을 전환하는 시스템을 제시한다. 이는 자원 조건에 따라 실행 방식이나 모델을 바꿀 동기를 뒷받침한다. 다만 전자는 경량화를 통한 대안이고 후자는 image semantic segmentation의 전환 사례이므로, 두 연구 자체가 SAM 2의 cross-model memory handoff 필요성을 입증하는 것은 아니다. [EdgeTAM, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_EdgeTAM_On-Device_Track_Anything_Model_CVPR_2025_paper.html), [ECSeg, SECON 2024](https://tanrui.github.io/pub/ECSeg-SECON.pdf).

상태를 가진 VOS 모델을 영상 중간에 교체하는 상황에서는, target에 과거 추적 정보를 어떻게 제공할지가 추가 문제가 된다. 전체 과거를 target으로 재처리하면 target-native 상태를 구성할 수 있지만 그에 따른 계산과 데이터 접근이 필요하다. 반대로 원래 prompt나 일부 source 마스크만 재인코딩하는 방식은 간단하지만, 누적 상태를 이용하는 방식과 정확도·비용 특성이 다를 수 있다. LLM의 Cross-Model KV Cache Transfer는 source의 내부 상태를 변환해 target의 과거 입력 재처리를 줄일 수 있다는 선행 사례를 제공한다. 그러나 VOS의 공간적 메모리, 객체별 정체성, 가림·재등장에는 별도의 상태 계약과 downstream 검증이 필요하다. [Cross-Model KV Cache Transfer, 2026 preprint](https://arxiv.org/abs/2608.03893).

특히 전환 직전에는 보이지 않다가 이후 재등장하는 객체는 최신 마스크만으로 전달하기 어려운 사례다. SAM2Long은 메모리와 예측 오류의 누적 문제를 다루고, MOSEv2와 LVOS는 사라짐·재등장과 장기적 어려움을 별도로 분석한다. 따라서 handoff의 효과는 전체 평균뿐 아니라 객체의 관측 이력과 전환 이후 사건을 구분해 평가해야 한다. [SAM2Long, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html), [MOSEv2](https://arxiv.org/abs/2508.05630), [LVOS v2](https://arxiv.org/abs/2404.19326).

본 연구는 이러한 전환 상황에서 source의 객체별 temporal state를 target이 사용할 수 있는 상태로 변환하는 nonlinear translator를 연구한다. SAM 2.1 Tiny→Large를 통제된 초기 모델 쌍으로 사용하고, 직접 상태 복사, 객체별 원래 prompt 및 마지막 유효 source 마스크의 재인코딩, 짧은 replay와 비교한다. 핵심 질문은 상태 tensor의 복원 정확도가 아니라, 과거 재처리를 줄이면서 전환 이후 분할 성능과 객체 재발견을 유지할 수 있는지이다. 이 효과는 실험으로 검증해야 할 가설이며, 특정 모델 쌍에서 translation이 항상 필요하거나 우월하다고 전제하지 않는다.

### 2.2 주장할 수 있는 것과 아직 입증하지 못한 것

| 근거 | 뒷받침하는 주장 | 뒷받침하지 않는 주장 |
|---|---|---|
| SAM 2 / XMem / Cutie | VOS에서 시간적·객체별 상태가 중요함 | 다른 모델에서도 그 상태가 그대로 호환됨 |
| EdgeTAM | 메모리 처리 비용과 on-device 효율이 중요한 문제임 | 반드시 모델 전환으로 해결해야 함 |
| ECSeg | 서로 다른 segmentation 모델 간 전환의 시스템 연구 사례가 있음 | SAM 2 handoff가 상용 현장에서 이미 필수임 |
| Cross-Model KV Cache Transfer | 내부 상태 변환으로 과거 재처리를 줄이는 방법론적 선례 | 동일한 효과가 VOS에서도 보장됨, nonlinear가 항상 최선임 |
| SAM2Long / MOSEv2 / LVOS | 오류 누적·재등장·장기 조건의 검증이 필요함 | 그 문제의 해결책이 반드시 cross-model translator임 |

**필요성 검증 순서:** (a) 모델을 바꿔야 할 구체적인 운영 조건을 정한다. (b) 같은 사례에서 source-only보다 target으로 바꿀 이익이 있는지 확인한다. (c) 강한 prompt/mask/replay 대안보다 translator의 정확도–지연–전송량 trade-off가 나은지 측정한다. 이 세 가지가 성립하지 않으면 “SAM 2 버전 사이 translator가 필요하다”는 결론을 내리지 않는다.

또한 본 연구의 nonlinear 범위는 연구 선택이다. 기존 Linear/Ridge 연구 중단을 성능 열세의 증거로 기술하지 않는다.

## 3. 수정 baseline 설계

### 3.1 공통 조건

- source가 프레임 `t`까지 처리한 뒤 전환하고, 정확도는 `t+1`부터 같은 미래 구간에서 평가한다.
- 평가 대상은 `t`까지 실제로 prompt로 등록한 동일한 객체 집합이다. 모든 객체가 영상 0번에서 등록됐다고 가정하지 않는다.
- 모든 방법에 동일한 과거 데이터 접근 권한을 부여하되, 실제 사용·보관·전송·재인코딩한 데이터와 비용을 따로 센다.
- 원래 task에서 제공한 prompt만 사용한다. switch 시점의 새 GT 마스크나 미래 GT를 초기화에 사용하지 않는다.
- 원래 prompt가 point/box이면 그대로 재사용하는 것이 기본이다. 이를 GT mask로 교체하는 것은 추가 정보를 준 다른 실험이다.
- 모델 가중치와 미래 추론 설정을 고정한다. target-native 결과는 참조값이지 평가 GT가 아니다.
- 아래 신규 방식은 **설계 완료·구현 검증 전**이다. 현재 single-object pilot runner의 동작과 구분한다.

### 3.2 핵심 비교군

| 비교군 | 전환 시 쓰는 데이터 | target 상태 구성 | 답하려는 질문 |
|---|---|---|---|
| Source-only | source의 기존 state | target으로 바꾸지 않음 | 애초에 전환의 이익이 있는가? |
| Target-native / Full Replay | 등록부터 전환까지의 RGB와 실제 prompt timeline | target이 과거를 순서대로 처리 | target이 과거를 직접 봤을 때의 참조 성능과 재처리 비용은? |
| Direct State Copy | source memory, pointer, presence 및 필요한 history/metadata | 연속값을 학습 없이 복사하고 target PE·상태 계약을 적용 | 번역 학습 없이도 충분한가? |
| Original-Prompt(s) Only | **객체별** 최초 prompt와 해당 RGB/실제 frame ID | anchor만 target에 재인코딩 | 원래 객체 지정 정보만으로 충분한가? |
| Last-Visible Source Mask | **객체별** 마지막 비어 있지 않은 source 예측 mask와 해당 RGB/실제 frame ID | 각 객체의 최신 관측을 target prompt로 재인코딩 | 최신 유효 관측 하나면 충분한가? |
| Original + Last-Visible | 앞의 두 anchor 집합 | 중복 frame/object는 정리하고 두 정보를 재인코딩 | 오래된 신뢰 가능한 지정과 최신 관측의 조합이면 충분한가? |
| Original-Prompt(s) + Replay-k | 객체별 원래 anchor와 전환 전 최근 RGB `k`장 | anchor 초기화 후 최근 구간 추론 | 모든 객체의 지정 정보를 보장한 짧은 replay면 충분한가? |
| Nonlinear Translator | source의 전달 대상 state와 metadata | 연속 state를 변환하고 target 계약으로 주입 | prompt 재인코딩/replay보다 좋은 trade-off가 가능한가? |

Source-only는 수학적 하한이 아니고 Target-native는 수학적 상한이 아니다. 다른 history/초기화가 target-native보다 높은 GT J&F를 낼 수 있다. Full Replay 비용 측정과 미리 계산해 둔 target-native 정확도 참조도 구분한다. target을 평소에도 병렬 실행했다면 그 상시 비용을 별도로 포함해야 한다.

Original과 Last-Visible은 따로 둔다. 둘을 처음부터 합치면 원래 prompt의 효과인지 최신 source 관측의 효과인지 구분할 수 없다. 결합군은 두 단독군을 대체하지 않는 강한 실용적 대안이다.

### 3.3 기존 비교군의 유지·명칭 정리

- **Last-Mask:** 정확히 `t`의 source 예측 mask와 RGB로 초기화한다. 객체별 mask가 비어 있으면 그대로 빈 정보를 준다. 몰래 last-visible로 대체하지 않는다.
- **Recent-Window Replay-k:** 현 runner의 Replay-k. `t-k+1`의 source 예측 mask를 prompt로 쓰고 `t`까지 재처리한다. 현재 구현은 single-object이며 `Replay-1=Last-Mask`다. 최근 관측에 의존하는 저비용 대안으로 유지하지만 complete multi-object handoff의 유일한 대안으로 삼지 않는다.
- **Empty-mask Reset proxy:** 현 Reset은 빈 mask로 객체 slot을 등록하는 구현이다. prompt 없는 target이 원하는 객체를 알아내는 일반적인 reset 방식이라고 표현하지 않는다.
- **Same-checkpoint export→inject:** 같은 checkpoint로 상태를 내보냈다가 복원하고 uninterrupted 추론과 비교하는 파이프라인 검증이다. 경쟁 모델이 아니다.
- **Translation + Short Replay:** 순수 translation의 효과를 먼저 분리한 뒤 hybrid 성능/비용을 추가 비교한다.
- **Source-history keyframe re-encoding:** translator가 오래된 history를 많이 이용한다면, 동일 history에서 고른 RGB+source mask를 target에 재인코딩하는 대조군도 추가한다. 이는 새 학습 없이 history 활용 자체의 효과를 검증한다.

### 3.4 재현 가능한 구현 규칙

1. **원래 prompt의 범위:** 기본 `Original`은 객체별 최초 prompt다. 이후 사용자 correction이 있는 실험에는 `All-Original-Prompts` 변형을 추가한다. Full Replay와 translator도 같은 실제 correction timeline을 사용한다.
2. **Last-visible의 선택:** `s_i = max{s ≤ t : source mask(i,s)가 비어 있지 않음}`으로 정의한다. GT-visible frame을 찾아 주지 않는다. 면적·presence 임계값을 쓰면 validation에서 미리 정하고 이름/설정에 남긴다. source가 다른 물체를 잘못 잡은 mask도 선택될 수 있다는 한계를 포함한다.
3. **선택 실패:** 유효 source mask가 한 번도 없으면 원래 prompt로 fallback하는 운영형을 명시하고 fallback 비율을 보고한다. 엄격한 last-visible-only 결과를 낼 경우 선택 실패를 제외하지 말고 별도 표시한다.
4. **시간 정보 보존:** 객체 A는 10번, B는 30번에서 등록됐다면 해당 RGB와 실제 frame index를 사용한다. 오래된 mask를 50번 RGB에 붙이거나 모든 anchor를 0번으로 바꾸지 않는다.
5. **Sparse reconstruction:** anchor와 최근 창 사이를 생략하면 full target-native 상태가 재현되는 것이 아니다. 최근 window 시작 때 필요한 output/PE/frame metadata를 target 정책에 맞게 구성하는 테스트가 선행돼야 한다.
6. **Replay 예산:** `k`는 최근 추가 RGB frame 수다. anchor까지 합친 unique RGB encode 횟수, 객체별 decoder/memory encode 횟수, 중복 제거 기준을 함께 기록한다. 객체 10개의 anchor 비용을 `k=1`에 숨기지 않는다.
7. **재인코딩에 필요한 자료:** mask만으로는 일반적으로 충분하지 않고 해당 과거 RGB가 필요하다. raw RGB를 평소 보관·검색하는 비용과 source 예측 history 캐시 유지 비용을 함께 센다. 로컬 파일이 있다는 이유로 원격 target에 무료로 있다고 가정하지 않는다.
8. **State 전달 범위:** forward-only에 필요한 state와 interactive history 전체를 구분해 고정한다. Direct와 Translator에 동일한 state export 범위를 적용한다.
9. **검증:** object ID 보존, 빈 예측/재등장, 서로 다른 최초 prompt frame, 오래된 anchor, pointer/PE/frame index, 객체 간 non-overlap 처리에 대한 테스트를 통과한 뒤 본 실험에 사용한다.

## 4. 객체 특성별 평가의 문헌 근거와 인용 가치

| 논문 | 실제 분석 방식 | 이번 연구에서의 인용 가치 |
|---|---|---|
| **MOSEv2 (2025 공개본)** | instance-sequence 속성 분석과 disappearance/reappearance 구간 분리 평가. §III-B/C, §IV-E | **최우선.** 부재 중 배경으로 예측하는 능력과 재등장 후 다시 찾는 능력을 구분할 직접 근거 |
| **LVOS v2 (2024 공개본; TPAMI 2025)** | long-term reappearance, cross-temporal confusion 등의 속성별 성능. §III, §IV-C/Table VII | **최우선.** 긴 관측 공백과 시간차를 두고 나타나는 유사 객체에 대한 분석 근거 |
| **SAM2Long (ICCV 2025)** | SAM 2의 오류 누적에 대응하는 memory-tree 방법과 장기 영상 실험 | **필수 보조.** 메모리 품질과 장기 추론의 연결 근거. switch-specific benchmark의 출처로 쓰지는 않음 |
| **Robust Promptable Video Object Segmentation (CVPR 2026)** | adverse condition과 시간에 따라 변하는 영상 열화에 대한 RobustPVOS benchmark | **조건부 추가.** 조명·날씨·영상 열화 축을 추가할 때 유용. 재등장 handoff의 직접 근거는 아님 |

원문: [MOSEv2](https://arxiv.org/html/2508.05630v1), [LVOS v2](https://arxiv.org/html/2404.19326v2), [SAM2Long](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html), [RobustPVOS 논문](https://arxiv.org/abs/2605.12006)·[공식 프로젝트](https://sohyun-l.github.io/RobustPVOS_project_page/). LVOS의 게재 정보는 [공식 저장소](https://github.com/LingyiHongfd/LVOS)의 인용 항목으로 확인했다.

주의할 점은 이 논문들의 속성이 곧바로 “모델 전환 직전/직후” 속성은 아니라는 것이다. **비디오의 객체별 난이도를 분석하는 문헌을 근거로 삼되, 객체–전환 시점 단위의 분석은 CMMT가 별도로 정의하는 실험 설계**라고 기술한다.

MOSEv2의 재등장 지표는 처음부터 보이던 구간을 제외하며, 수정 경계지표 `Ḟ`와 구간별 집계 규칙을 사용한다. 기존 CMMT의 모든 GT-visible frame 평균을 그 공식 재등장 지표와 동일하다고 쓰면 안 된다. LVOS의 LRA 기준에는 100-frame 공백이 등장하지만 데이터 sampling rate가 다르면 동일한 시간 길이가 아니므로 실제 timestamp/초를 함께 보고한다.

## 5. CMMT에 적용할 객체–전환 조건별 평가

### 5.1 평가 단위

단위는 비디오 하나가 아니라 **`(video_id, object_id, switch_frame)`**다. 같은 비디오라도 객체 A는 계속 보이고 B는 한참 뒤에 재등장할 수 있으며, switch를 어디에 두는지에 따라서도 조건이 바뀐다.

전체 manifest의 모든 방법을 같은 사례에 실행한다. 그 결과를 아래 조건으로 나눠 해석한다. 미래 GT는 오직 평가와 strata 분류에만 사용하고, 실행할 방법이나 전달할 frame을 고르는 데 사용하지 않는다.

### 5.2 기본 분류

`최근`의 길이는 보고할 고정 window `w`로 정의한다. replay 길이 `k`를 바꿀 때 같은 strata의 사례가 바뀌지 않도록 기본 `w`는 고정하고, 필요하면 `k`별 coverage를 별도 분석한다.

| 조건 | 정의 | 주요 질문 |
|---|---|---|
| 전환 시 보임 | GT에서 객체가 `t`에 보임 | 강한 최신 관측이 있을 때도 translation이 가치 있는가? |
| 최근까지 보이다가 부재 | `t`에는 없지만 최근 `w`프레임 안에 보인 기록이 있음 | 짧은 관측 공백은 replay/last-visible로 충분한가? |
| 오래 부재하다 재등장 | 최근 `w`에는 없고 그 이전에는 보였으며, 평가 구간에서 다시 나타남 | 원래 prompt·오래된 source 관측·누적 state 중 무엇이 도움이 되는가? |
| 평가 구간에서 계속 부재 | switch 이후 정해진 평가 구간 전체에 GT 객체가 없음 | 없는 객체를 잘못 만들어 내는 false positive를 억제하는가? |

첫 세 조건에는 이후 재등장 여부를 추가 표시한다. 마지막 조건은 과거 조건과 겹칠 수 있는 별도 future-event 축이다. 서로 겹치는 속성별 점수를 합산해 전체 점수를 만들지 않는다. 필요하면 과거 상태×미래 사건의 교차표로 완전한 partition을 만든다.

**GT와 source의 관측을 분리한다.** 예를 들어 실제 객체는 `t`에 보이지만 source가 놓쳤다면, 이는 진짜 부재와 다른 실패다. `GT last-visible age`, `source-predicted last-visible age`, 선택된 anchor가 정답 객체를 얼마나 잘 잡았는지를 각각 기록한다.

추가 축으로 객체 크기, 외형 변화, 유사 객체, 빠른 움직임, shot change를 사용한다. 빈 GT mask만 보고 “완전 가림”인지 “화면 밖”인지 단정하지 않는다. 공식 속성 annotation 또는 수동 검수가 필요하다. annotation 누락도 객체 부재로 취급하지 않는다.

### 5.3 지표와 보고 방식

- **분할 품질:** 공식 전체 J&F와 J/F를 유지하고, GT-visible 및 재등장 구간 결과를 함께 보고한다. MOSEv2 공식 `J&Ḟ_d`, `J&Ḟ_r`는 공식 구현·집계를 재현한 경우에만 그 이름을 쓴다.
- **부재 중 오류:** GT-absent frame의 false-positive 면적/비율과 잘못된 객체 예측을 별도 보고한다. 계속 빈 마스크만 예측해 전체 점수가 좋아 보이는 상황을 점검한다.
- **재등장 회복:** 첫 재등장 뒤 1/5/20 observation의 J&F, 정해진 기준에 도달하기까지의 시간, 끝까지 회복하지 못한 비율을 보고한다. 이후 다시 부재가 되는 경우의 처리와 관측 간격도 명시한다.
- **참조군 대비:** source-only와 target-native 각각에 대한 차이를 보고한다. target-native가 실패한 경우까지 target-relative recovery만으로 판단하지 않는다. GT 기준 재발견도 필요하다.
- **정체성:** 기존 single-object `identity-loss proxy`를 true multi-object ID switch라고 쓰지 않는다. 다객체 identity metric은 고정 object ID와 GT 대응, 누락/교환의 판정 규칙을 별도로 정의한다.
- **효율:** handoff latency, 첫 미래 frame 결과까지의 지연, unique RGB 및 object decode 횟수, cache 보관량, 실제 전송 bytes, peak VRAM을 함께 보고한다. 전체 prefix 유지 비용과 전환 순간 비용을 분리한다.
- **통계:** 각 stratum의 비디오·객체·switch 수를 밝히고, 같은 비디오의 여러 객체/전환을 독립 표본으로 과장하지 않는다. video-clustered bootstrap CI를 사용한다.

주장 예시는 “모든 영상에서 translator가 우수하다”가 아니라, 실험으로 확인된 경우에 한해 “장기 부재 후 재등장 조건에서 강한 anchor/replay 비교군보다 같은 비용에서 복구율이 높다”가 된다. 유리한 subset 결과와 전체 workload 평균/비용은 함께 제시한다.

## 6. 실행 순서와 현재 위치

```text
기존 single-object pilot·state 수집/MLP 작업
    ↓
필요성 근거 + baseline/평가 정의 보완  ← 이번 문서화 범위
    ↓
객체별 anchor·부재·다객체 state 계약 및 신규 runner 검증
    ↓
동일 manifest의 Source-only / Target-native / 강한 재인코딩 비교
    ↓
Nonlinear 학습·held-out 전체 및 조건별 평가
    ↓
MOSEv2/LVOS 장기·재등장 검증 + 정확도–비용 분석
    ↓
Tiny→Large 결과 정리 → 필요성 확인 후 반복 전환/이질 구조 확장
```

기존 실험 결과는 보존한다. 새 설계로 재평가하지 않은 pilot을 새 다객체 비교군·공식 재등장 지표의 결과로 재명명하지 않는다. 이번 작업은 문헌·코드 근거 확인과 문서 수정이며, GPU 실험이나 신규 runner 구현은 수행하지 않았다.
