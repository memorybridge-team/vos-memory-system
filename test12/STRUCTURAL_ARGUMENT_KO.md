> 게시 패키지 안내: 재현 명령과 경로는 [README.md](README.md)를 따른다. 아래 내용은 실험 당시 설명이다. 원본 소스는 `provenance/`, 실행용 benchmark 코드는 `runtime/`에 보존했다.

# Base+ Native와 Full Replay의 조건부 동등성

## 먼저 구분할 주장

1. 알고리즘 동등성: 같은 Base+ 가중치, 최초 프롬프트, RGB 프레임 순서, 추론 설정이면 Native와 Full Replay는 같은 상태 전이열을 계산한다.
2. 실행의 정확한 일치: 같은 환경의 두 독립 실행에서 모든 출력 텐서의 값과 마스크가 정확히 일치한다.
3. 수치적 근사 동등성: 독립 실행 차이가 존재하더라도 Native–Full Replay 차이가 같은 방법의 반복 오차 범위와 비슷하다.

첫째 주장을 코드 구조와 상태 전이로 설명할 수 있다. 둘째는 실제 장치와 구현에 따라 검증해야 한다. 셋째는 반복 자료와 정한 허용 기준이 필요하며, 몇 개의 영상에서 평균이 비슷하다는 사실만으로 통계적 동등성을 증명할 수 없다.

## 상태 전이의 정의

`theta`는 고정한 Base+ 가중치와 추론 설정, `I_t`는 t번 RGB, `P`는 최초 마스크 프롬프트다. `S_t`는 t번 추적 이후의 추론 상태, `Y_t`는 해당 출력이다.

```text
S_init = Initialize(video, theta)
S_prompt = AddMask(S_init, frame=0, mask=P)
(S_t, Y_t) = TrackOneFrame(S_previous, I_t, theta)
```

상태에는 조건/비조건 기억, `maskmem_features`, `maskmem_pos_enc`, `pred_masks`, `obj_ptr`, 객체 존재 로짓, 프롬프트 입력, 임시 출력, 추적 메타데이터, 객체 매핑, 캐시와 상수가 포함된다. 현재 `Session`은 `MEMORY_WINDOW=16`보다 오래된 비조건 출력을 각 프레임에서 같은 규칙으로 제거한다. 다른 전환 방법처럼 Small 상태를 주입하는 단계는 Full Replay에 없다.

## 코드에서 확인하는 동일성

- `baseline/no_handoff.py::full_replay`는 `session.add_prompt(prompt_frame, prompt_mask)`를 실행하고 최초 프레임 번호를 반환한다. 동일한 마스크를 주는 Native 경로와 같은 작업이다.
- `evaluation/evaluate_video.py::_run_base`는 동일한 `session.encode_prompts()` 뒤에 `session.track(track_from, end)`를 호출한다. `keep_after`는 어떤 프레임을 채점해 저장할지 결정하며, 영상 전방향 추론 시작이나 입력을 변경하지 않는다.
- `model/sam2_runner.py::Session.track`는 같은 predictor의 `propagate_in_video`를 호출하고 같은 기억 삭제 규칙을 적용한다.
- 로컬 SAM2 `propagate_in_video`에서 `max_frame_num_to_track`는 반복의 종료 프레임만 결정한다. `_run_single_frame_inference`에 전환 비율이나 구간 길이를 전달하지 않는다. 단, 상태의 `num_frames`는 모든 경로에서 원래 입력 구간 전체 길이로 유지해야 한다.
- 새 iterator는 `propagate_in_video_preflight`를 다시 호출한다. 새 프롬프트/임시 출력이 없고 해당 설정에서 전처리가 상태를 변경하지 않는다면, 구간 경계는 상태 전이열에 영향을 주지 않는다. 이 조건을 추측만 하지 않고 전환 직전 전체 상태와 추가 preflight 직후 전체 상태로 검사한다.

현재 벤치마크는 Native를 별도 알고리즘으로 실행하지 않고 Full Replay 실행을 Native reference로 저장한 뒤 점수와 기억을 재사용한다. 이는 정의를 일관되게 유지하지만 독립 실행 재현성 증거가 아니다.

## 귀납 논증

다음 전제가 모두 참이라고 가정한다.

- 초기화, 가중치, 프롬프트, RGB, 총 프레임 수, 설정, 객체 매핑이 같다.
- `TrackOneFrame`과 기억 삭제 규칙이 같은 함수이며 결정적으로 계산된다.
- 구간 재시작 시 추가 preflight가 추론 상태를 변경하지 않는다.

초기 상태가 같으므로 첫 추적 입력이 같다. t−1까지 상태가 같다고 가정하면, t번 입력과 전이 함수가 같아 t번 상태와 출력도 같다. 따라서 모든 t에서 `S_t_native = S_t_replay`, `Y_t_native = Y_t_replay`다. 전환 s에서 양쪽 상태가 같으므로 s+1 이후에도 같은 논증이 적용된다.

이는 위 전제를 둔 알고리즘 논증이다. 모든 가능한 영상과 플랫폼을 다루는 기계 검증이나 비결정적인 커널까지 포함하는 무조건적인 증명은 아니다.

## 이번 실험이 검사하는 전제

- DAVIS 서로 다른 영상 3개, 각 33프레임과 최초 GT 프롬프트. 전환 8/16/24.
- 모든 입력 RGB·GT·최초 마스크, 체크포인트, 소스의 SHA256 기록.
- Native 두 실행, 독립 연속 Full Replay, 각 전환에서 나누는 Full Replay 세 실행. 모두 새 추론 상태를 만든다.
- 모델 가중치의 실행 전후 해시를 비교한다.
- 모든 프레임의 원본 해상도 로짓, 이진 마스크, 저해상도 로짓, 기억 특징/위치 부호화, 포인터, 존재 로짓을 비교한다. 원소별 최대/평균 절대 오차, RMSE, 정확한 일치, 고정 허용 오차 `rtol=1e-5, atol=1e-6` 여부를 남긴다.
- 전환 시점에는 입력 reader를 제외한 `inference_state` 전체를 재귀 비교한다. 입력 reader는 동일 RGB 해시 및 순서로 별도 검사한다. 텐서 dtype·shape, 딕셔너리 키, 리스트 길이와 비텐서 값도 검사한다.
- 전환에서 추가 preflight 전후 상태, 단일 프레임 추론 호출과 기억 인코딩 요청 플래그를 검사한다. `_run_memory_encoder` 래퍼 호출 기록은 최초 프롬프트 preflight만 포함한다. 후속 프레임은 SAM2Base의 `_encode_new_memory` 경로를 사용하므로 전체 기억 인코더 모듈 호출은 직접 계측하지 않았다. 후속 기억 텐서의 값은 모든 프레임에서 비교했다.
- GT J/F/J&F, Native 대비 마스크 IoU와 J&F 차이를 프레임별로 기록한다. 객체가 사라진 경우도 원래 프레임 집합에 그대로 포함한다.

CPU에는 `torch.use_deterministic_algorithms(True)`를 사용하고, MPS 기본 실행은 Native 반복 차이를 대조로 삼는다. 이 실행은 성능 측정이 아니며 계측 시간을 모델 지연으로 사용하지 않는다.

## 불일치가 나왔을 때의 판단

Native 반복 자체가 다르면 Native–Full Replay 차이를 바로 전환 알고리즘 차이로 분류할 수 없다. 재생 구간에서 첫 차이가 나는지, 추가 preflight가 상태를 변경하는지, 호출 순서가 다른지, Full Replay 차이가 Native 반복보다 커지는지 순서대로 확인한다. 출력이 거의 같은 것과 모든 기억 텐서가 같은 것은 별도로 보고한다.

독립 실행 오차는 반드시 생기는 것이 아니다. 고정한 실행 환경에서 결정성을 확보하면 정확한 일치가 가능하다. PyTorch는 서로 다른 버전과 플랫폼 또는 CPU/GPU 사이의 완전 재현성을 보장하지 않으며, seed 고정과 결정적 연산 설정을 각각 설명한다. [PyTorch 2.8 Reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html).

현재 로컬 SAM2 소스는 이전 RunPod 소스와 다르고 CUDA BF16 경로도 실행하지 않는다. 이번 검증으로 RunPod 전체 벤치마크의 독립 실행 동등성을 확인했다고 표현해서는 안 된다.
