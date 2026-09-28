# Task 06 correction routing and repeated-switch closure

## 목적

SAM 2.1 Base+의 v1.1 최소 handoff 계약이 다음 세 상황에서도 원래 Base+ 실행과 같은 후속 결과를 만드는지 검증했다.

1. 전환 이후 frame에 correction이 들어오는 경우
2. 전환 이전 frame에 correction이 들어와 Target replay가 필요한 경우
3. 같은 영상에서 handoff를 두 번 연속 수행하는 경우

이 검증은 **same-checkpoint runtime 구현 정확성**을 확인한다. Small→Base+ nonlinear translator의 성능 실험은 아니다.

## 환경

- GPU: NVIDIA RTX A4000 16 GB
- SAM 2 upstream commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- checkpoint: `sam2.1_hiera_base_plus.pt`
- checkpoint SHA-256: `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`
- dataset case: DAVIS 2017 validation `walking`, object 1
- seed: 7
- 전체 회귀검사: `57 passed`

## 결과

| Gate | 설정 | 결과 | 비용 확인 |
|---|---|---|---|
| 전환 이후 correction | switch 10, correction 20 | frame 20–71, 52 frames: MSE 0, max error 0, binary IoU 1.0 | history replay 0, injection backbone call 0 |
| 전환 이전 correction | switch 20, correction 10 | Target가 prompt anchor 0부터 frame 20까지 21 frames replay; frame 21–71, 51 frames exact | replay 경로가 명시적으로 선택됨 |
| 반복 handoff | switch 10 → switch 20 | 첫 전환 뒤 61 frames, 두 번째 전환 뒤 51 frames: 모두 MSE 0, max error 0, binary IoU 1.0 | 두 injection 모두 backbone call 0 |

원본 JSON:

- [`post_switch_correction.json`](post_switch_correction.json)
- [`pre_switch_correction_replay.json`](pre_switch_correction_replay.json)
- [`repeated_switch.json`](repeated_switch.json)

## 반복 handoff에서 발견한 계약 결함과 수정

첫 반복 실행은 두 번째 export에서 `object_score_logits`가 없다는 이유로 중단됐다. v1.1 계약은 이 값을 translator 입력·출력이나 Target history에 넣지 않는 **진단용 값**으로 정의하므로, injected history에서 이 값이 없는 것이 정상이다. 그런데 exporter가 모든 record에서 이를 필수로 요구해 최소 계약이 반복 handoff에 닫혀 있지 않았다.

수정 후 exporter는 continuation에 필요한 `maskmem_features`와 `obj_ptr`만 필수로 검사한다. presence score가 없으면 canonical diagnostic slot은 0으로 유지하고 `missing_presence_records` metadata에 해당 record를 기록한다. handoff payload나 Target history에 score를 다시 추가하지 않았으므로 v1.1 최소 계약과 전송량 정의는 바뀌지 않는다.

## 해석과 범위

- 전환 이후 correction은 현재 RGB와 correction prompt를 Target이 직접 처리하므로 과거 replay가 필요 없다.
- 전환 이전 correction은 translated history를 부분 수정하지 않고 실제 RGB·prompt timeline을 이용해 안전한 prompt anchor부터 Target history를 재생성한다.
- 동일 checkpoint 반복 handoff는 minimal history를 다시 export해도 정보 계약이 유지됨을 확인했다.
- correction mask는 runtime 경로의 정확성을 검사하기 위해 명시적으로 제공한 interaction input이다. 공개 benchmark에 없는 correction을 GT에서 임의 합성해 성능 비교에 사용하는 것이 아니다. Task 03 원칙대로 correction benchmark는 별도 interaction manifest가 있을 때 모든 비교군에 동일하게 적용한다.

기존 단일 객체, 다객체·late prompt, 부재·재등장, Small→Base+ Direct Copy 결과와 이번 correction·반복 handoff 결과를 합쳐 Task 06의 runtime 구현 gate를 완료로 판정한다.
