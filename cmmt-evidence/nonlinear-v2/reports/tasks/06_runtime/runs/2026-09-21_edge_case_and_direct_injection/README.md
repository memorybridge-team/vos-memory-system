# Task 06 edge-case self-injection and Direct Copy pilot

> Date: 2026-09-21 KST
> Official SAM 2 revision: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
> Dataset: DAVIS 2017
> Device: NVIDIA RTX 2000 Ada Generation 16 GB

## 결과 요약

| Gate | 조건 | 후속 비교 | 과거 backbone replay | 결과 |
|---|---|---:|---:|---|
| 다객체·late prompt self-injection | Base+→Base+, `bike-packing`, object 1@frame 0, object 2@frame 10, switch 20 | 48 frames | 0 calls | mean MSE `0`, max error `0`, mean binary IoU `1.0` |
| 부재·재등장 self-injection | Base+→Base+, `india`, object 3@frame 0, switch 35 | 45 frames | 0 calls | mean MSE `0`, max error `0`, mean binary IoU `1.0` |
| cross-model Direct Copy | Small→Base+, `walking`, object 1@frame 0, switch 10 | 61 frames | 0 calls | mean binary IoU `0.0`; target-native와 일치 실패 |

다객체 실험은 switch 전에 서로 다른 시점에 등록된 두 객체의 conditioning 및
non-conditioning history 32개 record를 새 Base+ predictor에 조립했다. 부재·재등장
실험은 object 3이 switch 부근에서 보이지 않았다가 다시 나타나는 구간을 사용했다.
두 경우 모두 native Base+와 injected Base+의 후속 logits가 exact였다.

Small→Base+ Direct Copy는 11개 history record를 구조적으로 주입했고 주입 중 과거
frame을 재처리하지 않았다. 그러나 Target-native state 대비 spatial-memory cosine은
`0.0211`, object-pointer cosine은 `-0.0220`이었고, 후속 61 frames의 binary IoU가
모두 `0`이었다. 이는 한 DAVIS case의 pilot이므로 전체 데이터셋 성능 결론은 아니지만,
동일 shape라는 이유만으로 Small state를 Base+에 직접 복사할 수 없으며 learned 또는
calibrated translation을 비교해야 한다는 강한 구현·방법론 근거다.

## 최소 handoff 계약 변경의 영향

이 pilot의 실제 Target history는 이미 `maskmem_features`, `obj_ptr`, Target-generated
`maskmem_pos_enc`만 사용했다. 따라서 영상 길이·해상도와 prompt/tracking dictionary를
CanonicalState에서 제거해도 위 후속-mask 결과의 해석은 바뀌지 않는다. 다만 당시 JSON의
`translated_bytes=5,778,476`은 예전 `continuous_bytes` 정의로 Source presence 44 bytes를
포함하고 discrete assembly metadata를 제외한 값이다. 현재 계약으로 같은 단일 객체·11
record payload를 계산하면 `5,778,641 bytes`이며, 이후 표와 실험은 새 `handoff_bytes()`만
사용한다. 기존 JSON은 실행 당시 원본 증거이므로 덮어쓰지 않는다.

계약 refactor 뒤 CPU unit test를 먼저 통과시키고, Task 06 완료 전 실제 checkpoint로
same-checkpoint smoke를 한 번 재실행한다. 이는 기존 과학적 pilot을 다시 하는 것이 아니라
제외한 metadata가 continuation에 관여하지 않음을 확인하는 회귀 검사다.

## 남은 Task 06 gate

- switch 이후 현재 frame prompt correction을 Target-native 방식으로 처리하는 경로
- switch 이전 frame correction 요청을 안전한 Target replay fallback으로 보내는 경로
- correction 뒤 새 Target history가 정상적으로 이어지는지 strict comparison
- 여러 영상·switch frame에서의 반복 검증

## 원본 결과

- [`multiobject_late_prompt.json`](multiobject_late_prompt.json)
- [`reappearance.json`](reappearance.json)
- [`small_to_base_direct.json`](small_to_base_direct.json)
