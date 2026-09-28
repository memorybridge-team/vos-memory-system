# Task 06 — v1.1 최소 Target history 구현 기록

## 목적

State Assembly Map과 I/O 계약 v1.1에 맞춰 Source의 과거 `pred_masks`와 `object_score_logits`를 Target `inference_state`에 주입하지 않고도 `switch_frame + 1`부터 이어갈 수 있는 최소 history를 조립한다.

## 공식 SAM 2 코드에서 확인한 사실

- 다음 frame의 memory attention은 과거 record에서 `maskmem_features`, `maskmem_pos_enc`, `obj_ptr`를 읽는다: [SAM2Base memory read](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam2_base.py#L538-L587).
- propagation이 이미 저장된 conditioning frame 자체를 다시 출력하면 해당 record의 `pred_masks`를 읽는다: [propagate_in_video](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L546-L576).
- 같은 과거 frame에 correction click을 추가하는 기본 refinement 경로도 기존 `pred_masks`를 읽는다: [add_new_points_or_box](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_video_predictor.py#L230-L259).

따라서 “mask/score 비주입”은 임의 삭제가 아니라 다음 실행 계약과 함께 성립한다.

```text
정상 handoff: frame t까지 최소 read-state 주입 → frame t+1부터 propagation
전환 이후 correction: Target이 이미 생성한 현재 record에서 처리
전환 이전 correction: 원본 RGB + prompt timeline으로 Target replay
```

## 로컬 구현

- `materialize_sam2_history`가 유효 record마다 `maskmem_features`, Target-generated `maskmem_pos_enc`, `obj_ptr`만 생성한다.
- `inject_sam2_canonical_state`가 위 세 필드만 Target 장치 정책에 맞춰 배치한다.
- Source `pred_masks`는 `CanonicalState.metadata.preserved_pred_masks`에 남지만 Target history에는 들어가지 않는다.
- Source `presence_logits`도 CanonicalState 진단값으로 남지만 Target history의 `object_score_logits`로 복원하지 않는다.
- 다객체 materialization과 injection test에 금지 필드 부재 assertion을 추가했다.

## 검증 상태

- `python -m compileall -q src tests`: 통과.
- `git diff --check`: 통과.
- 로컬 Python 환경에는 `torch`와 `pytest`가 없어 unit test 전체 실행은 보류했다.
- 실제 Small/Base+ checkpoint runtime, 다객체, late prompt, 부재/재등장, 전환 전·후 correction은 미검증이다.

## 다음 gate

GPU 환경에서 기존 Base+→Base+ self-injection을 `switch_frame + 1` 시작으로 다시 실행해 native/injected mask 일치를 확인한다. 이후 다객체와 correction router/replay를 구현·검증한다. 이 gate가 통과하기 전 Task 06은 `In Progress`다.
