# Small → Base+ 상태 I/O 계약

> 상태: **FROZEN v1.1 — 2026-09-21**
>
> 계약 ID: `cmmt.small_to_base_plus.io.v1.1`
>
> 근거: 정적 config/code, 실제 Small/Base+ runtime inventory, paired dump, fail-closed validator
>
> 단일 객체 Base+ self-injection과 확장 continuation 검증은 이 계약을 소비하는 task 06의 증거다.
> 기준 upstream: Meta SAM 2 commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`
>
> 구현·검증 추적: [GitHub Issue #5](https://github.com/memorybridge-team/vos-memory-translator-nonlinear/issues/5)

## 결론

SAM 2.1 Small과 Base+는 backbone 크기는 다르지만 video-memory 경계의 기본 shape가 같다. 따라서 첫 nonlinear translator는 크기 변환기가 아니라 **같은 shape 안의 표현 의미를 Small 공간에서 Base+ 공간으로 바꾸는 component-wise mapper**로 정의한다.

Translator가 학습할 연속 입력·출력은 `spatial_memory`, `object_pointer` 두 종류다. `presence_logits`는 Source 진단 기록으로만 보존하며 Target history에 주입하지 않는다. Frame 번호, 객체 ID, conditioning 여부, record 순서와 validity는 학습하지 않고 정확히 복사한다. Spatial positional encoding은 source 것을 번역하거나 복사하지 않고 Base+가 다시 생성한다.

연구 단계에서는 Source와 Target이 같은 영상, 같은 frame 순서, 같은 preprocessing과 switch 시점을 사용한다고 실험 manifest에서 고정한다. 따라서 `num_frames`, `video_height`, `video_width`와 영상 fingerprint는 memory handoff payload 및 translator API에 포함하지 않는다. 앞의 세 값은 Target runtime이 자신의 영상에서 산출하고, hash 기반 영상 식별은 향후 분산 서비스 handoff protocol의 선택적 책임으로 분리한다.

## 1. 확인된 모델 경계

| 항목 | Small | Base+ | handoff 판단 |
|---|---:|---:|---|
| backbone 시작 차원 | 96 | 112 | translator 입력이 아님 |
| image neck / memory attention `d_model` | 256 | 256 | object pointer 경계와 연결 |
| memory encoder 출력 channel | 64 | 64 | spatial mapper 입력·출력 |
| 기본 memory grid | 64×64 | 64×64 | 1024 입력과 stride 16 기준 |
| object pointer 차원 | 256 | 256 | pointer mapper 입력·출력 |
| `num_maskmem` | 7 | 7 | 한 번의 memory read 범위; 전체 저장 history 상한이 아님 |
| `max_obj_ptrs_in_encoder` | 16 | 16 | 시간축 pointer read 범위; 객체 수 제한이 아님 |

위 표는 pinned config와 `SAM2Base` 생성 코드의 정적 계약이다. 2026-09-20 DAVIS `walking`, object 1, switch frame 10의 실제 checkpoint runtime에서도 Small과 Base+ 모두 spatial memory `[1,1,11,64,64,64]` bfloat16, object pointer `[1,1,11,256]` float32, presence logits `[1,1,11,1]` float32로 확인했다. 이 한 사례의 shape·dtype 일치는 경계를 확인하지만 표현 의미의 동일성을 뜻하지 않는다.

## 2. CanonicalState의 입력과 처리 정책

| Canonical field | 대표 shape | 출처 | 처리 정책 | Target 조립 위치 |
|---|---|---|---|---|
| `spatial_memory` | `[B,O,K,64,64,64]` | `maskmem_features` | nonlinear translation | `maskmem_features` |
| `object_pointer` | `[B,O,K,256]` | `obj_ptr` | 별도 nonlinear translation | `obj_ptr` |
| `presence_logits` | `[B,O,K,1]` | `object_score_logits` | Source 진단 기록만 보존; learned/direct/calibration 출력 및 Target history 주입에서 제외 | CMMT diagnostic sidecar |
| `frame_indices` | `[B,O,K]` | history dict key | 정확히 복사 | cond/non-cond frame key |
| `slot_order` | `[B,O,K]` | canonical record 순서 | 정확히 복사·검증 | materialization 순서 |
| `is_conditioning` | `[B,O,K]` | cond/non-cond dictionary | 정확히 복사 | history dictionary 선택 |
| `validity` | `[B,O,K]` | padding 여부 | 정확히 복사·loss mask | 유효 record만 조립 |
| `object_ids` | 길이 `O` | predictor registry | 정확히 복사 | `obj_id_to_idx`, `obj_idx_to_id` |
| `switch_frame` | scalar | handoff 시점 | 정확히 복사·pair 검증 | continuation 시작점 |

`B`는 현재 injection 구현에서 1, `O`는 prompt로 등록한 객체 수, `K`는 객체별 저장 history를 맞추기 위한 padded record 축이다. `K=7`로 고정하지 않는다. Predictor가 저장한 전체 history와 한 frame에서 실제 attention이 읽는 memory 수는 다르기 때문이다.

Export 직전에는 Source가 `non_blocking=True`로 GPU→CPU offload한 최신 memory가 완전히 도착하도록 producer CUDA device를 한 번 동기화한다. 그 다음 Injection 직전에는 Target runtime 정책에 따라 translated `spatial_memory`를 Target `storage_device`로, translated `object_pointer`와 Target-generated PE를 Target compute device로 이동한다. Source `pred_masks`와 `presence_logits`는 Target runtime에 이동·주입하지 않는다. Source device 문자열은 provenance일 뿐 Target 장치 설정을 덮어쓰지 않는다. Runtime에서 확인한 dtype은 spatial `bfloat16`, pointer/presence `float32`이며 translator 내부 정밀도와 최종 cast는 학습 config에 기록한다.

## 3. Translator에 넣지 않는 상태

다음 데이터는 handoff에는 필요하지만 learned tensor 입력으로 사용하지 않는다.

- `maskmem_pos_enc`: Base+ memory encoder의 position module로 grid에 맞춰 재생성한다.
- temporal position: `frame_indices`와 현재 frame의 차이로 Base+가 자체 정책에 따라 만든다.
- `pred_masks`: source의 low-resolution logits를 CMMT 외부 표시 archive로만 보존한다. Target history나 decoder refinement 입력에는 주입하지 않는다.
- `presence_logits`/`object_score_logits`: source 진단 기록으로만 보존한다. Target은 새 frame 또는 correction replay에서 score를 새로 계산한다.
- original point/mask prompts와 `frames_tracked_per_obj`: CanonicalState와 handoff payload에 넣지 않는다. 전환 이전 correction에 필요한 실제 prompt timeline은 dataset/interaction manifest가 별도 보존하며, 원본 RGB와 함께 Target replay 입력으로 사용한다.
- `cached_features`: 과거 RGB backbone feature를 전달하지 않는다. Fresh target runtime의 cache는 비운다.
- `temp_output_dict_per_obj`: 미완성 상호작용 상태는 전달하지 않고 target에서 빈 dictionary로 시작한다.
- `num_frames`, `video_height`, `video_width`: Target runtime이 같은 영상에서 직접 산출한다. 로컬 cache·loader가 dataset 무결성 assertion에 사용할 수는 있지만 연구 handoff payload와 전송 bytes에서는 제외한다.
- `video_fingerprint`, `prefix_fingerprint`: 연구 단계에서는 동일 영상을 manifest로 보장하므로 CanonicalState와 translator API에 추가하지 않는다. 장치·서버 간 서비스화에서 필요하면 model state 밖의 handoff envelope로 구현한다.

## 4. Paired-state 정렬 규칙

학습 pair `(Small state, Base+ native state)`는 다음이 모두 같아야 한다.

1. 실험 manifest가 지정한 동일 video, frame 순서와 preprocessing
2. prompt timeline과 object ID
3. switch frame
4. `[B,O,K]` record 축
5. `frame_indices`, `slot_order`, `is_conditioning`, `validity`

Continuous channel/grid/pointer 차원은 model pair에 따라 달라도 된다. 반대로 위 이산 계약이 다르면 tensor shape가 맞더라도 서로 다른 시간·객체 record를 학습시키는 것이므로 fail closed한다. 이를 위해 `validate_paired_state_contract`가 padding, slot, switch mismatch까지 거부하도록 구현했다. 현재 로컬 paired-cache validator가 frame 수·해상도도 검사하는 것은 dataset 생성 오류를 찾기 위한 실험 assertion이며, 해당 값을 runtime handoff payload로 정의한다는 뜻은 아니다.

## 5. 현재 구현과 검증 상태

| 항목 | 상태 |
|---|---|
| Canonical export·history materialization | 구현됨 |
| Target PE 재생성 | 구현됨 |
| object registry 복원, prompt/tracking dictionary 초기화 | 구현됨 — `object_ids`로 registry를 만들고 interaction dictionary는 빈 값으로 시작 |
| v1.1의 과거 mask/score 비주입 정책 | Base+ 단일 객체, 다객체·late prompt, 부재·재등장, correction, 반복 handoff strict round-trip 통과 |
| 비동기 GPU→CPU export 안정성 | export 경계 CUDA 동기화와 회귀 test, 주입 전 11-record exact parity로 확인 |
| Pair discrete timeline validator | 구현·CPU unit test 추가 |
| Base+ checkpoint same-model export→inject | DAVIS `walking`, object 1, switch 10 통과 |
| Small/Base+ 실제 runtime shape inventory | 단일 paired case 확인 |
| 다객체·late prompt·부재/재등장 continuation closure | task 06 실제 checkpoint strict round-trip 통과 |
| prompt correction 뒤 continuation closure | 전환 후 correction은 no-replay, 전환 전 correction은 prompt anchor replay로 exact 통과 |
| Small→Base+ paired-state 예시 dump | 생성·checksum 기록 완료 |

## 6. GPU 재개 시 첫 실행

2026-09-20에 `scripts/runpod_preflight.sh`로 revision·checkpoint·storage를 확인한 뒤 `scripts/runpod_base_plus_roundtrip.sh`를 실행했다. DAVIS `walking`, object 1, switch frame 10에서 이후 61 frames의 native/injected 결과가 binary IoU 1.0, MSE 0.0, max absolute error 0.0이었고 injection 중 과거 backbone 호출은 0회였다. 원본 증거는 [`reports/runtime/2026-09-20_base_plus_self_injection/`](../../reports/runtime/2026-09-20_base_plus_self_injection/)에 있다.

이 결과로 단일 객체·첫 frame prompt의 실행 경계는 통과했다. 같은 조건의 Small/Base+ paired example도 생성해 두 모델의 shape·dtype·timeline 일치를 확인했다. `.pt` cache는 166,882,485 bytes이므로 Git에는 checksum만 남기고 RunPod network volume에 보존한다. 상세 보고서는 [`reports/runtime/2026-09-20_small_base_runtime_inventory/`](../../reports/runtime/2026-09-20_small_base_runtime_inventory/)에 있다.

2026-09-21 v1.1 최소 history로 다시 검증하는 과정에서 최신 CPU-offloaded
`maskmem_features`를 export 완료 전에 읽는 race condition을 발견했다. Export
경계에 CUDA 동기화를 추가한 뒤 주입 직전 11개 record의 세 read-state 필드와
후속 61개 frame의 logits가 모두 exact가 됐다. 원인·실패 결과·수정 후 증거는
[`reports/runtime/2026-09-21_v1_1_base_plus_self_injection_after_sync/`](../../reports/runtime/2026-09-21_v1_1_base_plus_self_injection_after_sync/)에 있다.

2026-09-21 추가 gate에서 DAVIS `bike-packing`의 두 객체를 frame 0과 10에 각각 등록하고 switch 20에서 Base+ state를 복원했다. 이후 48 frames가 mean MSE `0`, max error `0`, binary IoU `1.0`이었으며 injection 중 과거 backbone 호출은 `0`이었다. DAVIS `india`의 부재·재등장 구간에서도 object 3, switch 35 이후 45 frames가 같은 exact 기준을 통과했다.

같은 날 Small→Base+ Direct Copy는 11개 history record를 replay 없이 정상 주입했지만, DAVIS `walking` switch 10 이후 61 frames에서 Base+-native 대비 mean binary IoU `0.0`이었다. spatial-memory cosine `0.0211`, object-pointer cosine `-0.0220`으로 표현 의미가 정렬되지 않았다. 이는 한 사례의 pilot이며 전체 성능 결론은 아니지만 cross-model injector의 기계적 동작과 learned/calibrated translation 필요성 검증을 분리해 보여 준다. 원본 증거는 [`reports/runtime/2026-09-21_task06_edge_case_and_direct_injection/`](../../reports/runtime/2026-09-21_task06_edge_case_and_direct_injection/)에 있다.

2026-09-23에는 전환 이후 correction, 전환 이전 correction replay, switch 10→20 반복 handoff를 추가 검증했다. 비교 구간의 모든 frame에서 MSE `0`, max error `0`, binary IoU `1.0`이었고 두 injection의 과거 backbone call은 모두 `0`이었다. 반복 handoff에서 injected record에 진단용 `object_score_logits`가 없으면 재-export가 실패하는 결함을 발견했다. exporter의 필수 continuation field를 `maskmem_features`와 `obj_ptr`로 바로잡고, 누락된 presence diagnostic은 `missing_presence_records` metadata에 기록하도록 수정했다. score를 handoff payload나 Target history에 다시 넣지는 않았으므로 v1.1 계약은 유지된다. 원본 증거는 [`reports/runtime/2026-09-23_task06_correction_and_repeated_switch/`](../../reports/runtime/2026-09-23_task06_correction_and_repeated_switch/)에 있다.

task 02는 실제 Small/Base+ inventory, paired dump, 필드 정책, fail-closed validator와 State Assembly Map을 기준으로 동결됐으며, 이를 소비하는 task 06 runtime 구현도 위 검증으로 완료됐다.

### v1.1 구현 안전 조건

고정한 공식 SAM 2 코드에서 다음 frame의 memory attention은 과거 record의 `maskmem_features`, `maskmem_pos_enc`, `obj_ptr`만 읽는다. 반면 이미 저장된 conditioning frame을 그대로 출력하거나 같은 과거 frame에 correction을 추가하는 기본 경로는 그 record의 `pred_masks`를 읽는다. 따라서 v1.1 injection은 반드시 `switch_frame + 1`에서 시작하며, `frame <= switch_frame` correction은 기본 refinement API를 직접 호출하지 않고 원본 RGB·prompt timeline을 이용한 Target replay로 보낸다. 이 조건을 어기는 실행은 지원되는 continuation이 아니다.

## 7. 동결 근거와 변경 관리

- Small/Base+ 실제 경계: spatial `[1,1,11,64,64,64]` bfloat16, pointer `[1,1,11,256]` float32, presence `[1,1,11,1]` float32
- paired cache: `166,882,485 bytes`, SHA-256 `5e9bca17217d522335acf80a454834cf2beab22d4dd8f6204fcd221d2bf1a5f0`
- validator: schema, switch frame, object/frame/slot/conditioning/validity 불일치를 fail closed; 영상 길이·크기·checksum 검사는 CanonicalState 밖 dataset/cache assertion으로만 사용
- 시각 계약: [`State Assembly Map`](../architecture/cmmt-state-assembly-map.html)
- runtime inventory: [`reports/runtime/2026-09-20_small_base_runtime_inventory/`](../../reports/runtime/2026-09-20_small_base_runtime_inventory/)
- v1.1 minimal-history strict round-trip: [`reports/runtime/2026-09-21_v1_1_base_plus_self_injection_after_sync/`](../../reports/runtime/2026-09-21_v1_1_base_plus_self_injection_after_sync/)
- task 06 edge cases and Direct Copy pilot: [`reports/runtime/2026-09-21_task06_edge_case_and_direct_injection/`](../../reports/runtime/2026-09-21_task06_edge_case_and_direct_injection/)
- task 06 correction and repeated switch closure: [`reports/runtime/2026-09-23_task06_correction_and_repeated_switch/`](../../reports/runtime/2026-09-23_task06_correction_and_repeated_switch/)

이 계약 이후 새 field, dtype, shape, copy/translate/regenerate 정책을 바꾸면 계약 버전을 올리고 다음을 함께 갱신한다: validator test, Map, example dump, checksum, 영향받는 paired-state shard 목록. Task 06의 edge-case 실패가 현재 계약의 누락을 드러낸 경우에도 조용히 덮어쓰지 않고 v1.1 이상의 변경 기록을 남긴다.

## 8. Task 02 완료 선언

- [x] continuous translator I/O와 shape·dtype 경계 고정
- [x] discrete metadata의 copy·alignment·validity 규칙 고정
- [x] Target PE 재생성과 runtime device 정책 고정
- [x] prompt timeline·표시 archive·correction replay 정책 고정
- [x] 실제 Small/Base+ inventory와 paired example/checksum 기록
- [x] fail-closed validator와 State Assembly Map 일치 검토

따라서 task 02의 계약 작업은 Done으로 판정한다. 이 계약을 소비하는 다객체·late prompt·재등장·correction·반복 handoff continuation은 task 06에서 실제 checkpoint로 검증을 마쳤다.
