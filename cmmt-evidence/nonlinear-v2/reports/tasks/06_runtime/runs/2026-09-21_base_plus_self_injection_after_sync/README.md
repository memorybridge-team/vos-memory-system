# Base+ v1.1 minimal-history self-injection

- Date: 2026-09-21 (Asia/Seoul)
- Project base commit: `226790d66b437273963c4c87d4fe4fa67c7eec18`
- SAM 2 commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Device: NVIDIA RTX 2000 Ada Generation, 16,380 MiB
- Dataset: DAVIS 2017 `walking`, object 1
- Switch frame: 10
- Seed: 7

## 무엇을 검증했는가

v1.1 주입 history에는 과거 `pred_masks`와 `object_score_logits`를 넣지 않고,
SAM 2 memory attention이 다음 frame부터 읽는 아래 세 필드만 넣었다.

- translated/copied `maskmem_features`
- Target-generated `maskmem_pos_enc`
- translated/copied `obj_ptr`

## 첫 실행의 실패와 원인

첫 실행은 mean binary IoU `0.9999767222`였지만 max absolute logit error가
`3.421504`여서 strict gate를 통과하지 못했다. History를 주입하기 직전에 비교한
결과, frame 0–9의 세 필드는 모두 exact였고 최신 frame 10의
`maskmem_features`만 MSE `0.6701763`, max error `4.15625`였다.

원인은 SAM 2가 `offload_state_to_cpu=True`에서 최신 memory를
`non_blocking=True`로 GPU에서 CPU로 복사하는 동안 exporter가 즉시 host tensor를
읽은 race condition이었다. 이는 최소 상태 계약에서 필드가 빠진 문제가 아니다.
Exporter가 CUDA producer device를 export 경계에서 한 번 동기화하도록 수정했다.

## 수정 후 결과

주입 직전 11개 record에서 세 필드가 모두 bit-exact였다. 전체 후속 61 frames
(11–71)의 결과도 다음과 같이 strict gate를 통과했다.

- Mean binary IoU: `1.0`
- Mean MSE: `0.0`
- Maximum absolute error: `0.0`
- Backbone calls during injection: `0`
- Wall time: `50.7926 s`
- Peak CUDA memory: `976,665,088 bytes`

## 재현 명령

```bash
bash scripts/runpod_base_plus_roundtrip.sh \
  /workspace/vos-memory-translator-nonlinear \
  /workspace/CMMT/data/DAVIS \
  walking 1 10 \
  2026-09-21_v1_1_base_plus_self_injection_after_sync
```

주입 전 history parity 진단은 `scripts/diagnose_same_checkpoint_history.py`로
재현한다. `report.json`은 전체 continuation metric, `history_diagnostic.json`은
주입 직전 record별 tensor parity를 담는다.

## 해석 한계

이 결과는 단일 객체·첫 frame prompt의 Base+→Base+ implementation gate다.
Small→Base+ translator 성능 증거가 아니다. Task 06 완료 전에는 다객체,
late-prompt, 부재 후 재등장, prompt correction 및 Small→Base+ target injection을
별도로 검증해야 한다.
