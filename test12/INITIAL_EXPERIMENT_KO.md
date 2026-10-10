> 게시 패키지 안내: 재현 명령과 경로는 [README.md](README.md)를 따른다. 아래 내용은 실험 당시 설명이다. 원본 소스는 `provenance/`, 실행용 benchmark 코드는 `runtime/`에 보존했다.

# test12: Base+ Native와 Full Replay의 독립 비교

## 목적

현재 벤치마크는 Base+ 전체 추적 결과를 Native reference로 저장하고, 같은 점수를 Full Replay에 재사용한다. `evaluation/evaluate_video.py`의 `native.pack(...)`, `Run(scores=reference_scores, ...)`, Full Replay 분기를 참조한다. 따라서 기존 표의 점수 일치는 독립 실행의 일치 여부를 검증하지 않는다.

이 테스트는 실제 Base+ 체크포인트로 별도 추론 상태를 생성한다. Native는 영상 처음부터 끝까지 하나의 추적 iterator로 실행한다. Full Replay는 새로운 세션에서 처음부터 전환 프레임까지 추적하고, 해당 iterator를 완전히 종료한 다음 전환 다음 프레임부터 새 iterator로 추적을 계속한다. 원래 벤치마크보다 엄격하게 추적 구간을 나누어 재시작까지 검증한다. 동일 모델 가중치는 재사용하지만 프레임 입력과 추론 상태는 각 세션에서 새로 생성하며, Native 예측이나 SQLite 결과는 Replay 입력으로 사용하지 않는다.

## 입력과 환경

- 현재 Mac의 Apple M2 GPU, PyTorch MPS, FP32 추론.
- SAM2.1 Hiera Base+ 실제 체크포인트. 원본 체크포인트와 predictor SHA256은 결과 JSON에 기록한다.
- SAM2 예제 `bedroom`의 실제 JPEG 33프레임, 프레임 번호 0~32, 객체 1개.
- 전환 프레임: 25% = 8, 50% = 16, 75% = 24. 전환 이후 비교 프레임은 각각 24, 16, 8개다.
- 초기 마스크는 `test11/fixture_cpu.npz`의 `initial`이다. 실제 Small 예측 마스크이며 GT가 아니다. Native와 Replay에는 동일 마스크를 제공한다.
- BF16 autocast와 프레임 prefetch를 끈다. SAM2는 기억 특징 저장 시 내부적으로 BF16을 사용한다. 비교에서는 BF16 값을 FP32로 손실 없이 변환한다.
- `test11/.venv`를 재사용한다. 패키지 버전은 `test11/requirements-lock.txt`를 참조한다.

## 비교 항목

모든 프레임에서 원본 해상도 이진 마스크, 저해상도 `pred_masks` 로짓, `maskmem_features`, `maskmem_pos_enc`, `obj_ptr`, `object_score_logits`를 비교한다. 각 필드의 정확한 값 일치 여부, 다른 원소 수, 최대 절대 오차를 기록한다. 모든 프레임이 빠짐없이 한 번씩 비교되었는지도 검사한다. 비교에 실패하면 종료 코드는 1이다.

## 실행

저장소 루트에서 실행한다. macOS 샌드박스에서는 MPS 접근 권한이 필요할 수 있다.

```bash
test11/.venv/bin/python -u test12/check_native_replay.py --device mps > test12/run_mps.log 2>&1
```

원시 결과: `results_mps.json`. 실행 로그: `run_mps.log`.

## 해석 범위

이 테스트는 동일한 Base+ 가중치와 프롬프트를 사용하는 전체 재생의 추적 동등성을 확인한다. Small→Base+ 상태 전이, Translator 성능 또는 GT 정확도 검증은 아니다. 한 영상과 객체 한 개로 확인하므로 전체 PUMaVOS/VOST와 CUDA BF16의 동등성을 직접 입증하지 않는다. 현재 SAM2 빌드에는 CUDA 후처리 확장이 없어 관련 구멍 채우기 후처리가 생략된다. 두 경로에 같은 조건이 적용된다.

실행 시간에는 CPU 복사 및 프레임별 비교가 포함되므로 성능 벤치마크 값으로 사용하지 않는다.

## 측정 결과 (2026-10-10)

MPS 본 실험은 33프레임을 사용했다. CPU 대조 실험과 MPS Native 반복 대조 실험은 원인 확인을 위해 동일 영상의 앞 9프레임을 사용했다. CPU에서는 전환 전 최소 프레임 기준을 명시적으로 2로 낮춰 전환 프레임 2/4/6을 검사했다. 이는 벤치마크 정식 평가 조건을 변경한 것이 아니라 이 폴더의 짧은 진단 실험에만 적용한 값이다.

| 장치 / 비교 | 전환 | 전환 이후 마스크 정확히 일치 | 프레임당 최대 다른 픽셀 | 저해상도 로짓 최대 절대 오차 |
| --- | --- | --- | --- | --- |
| MPS Native vs Full Replay | 25%, 프레임 8 | 0/24 | 214 | 11.933901 |
| MPS Native vs Full Replay | 50%, 프레임 16 | 0/16 | 301 | 12.766500 |
| MPS Native vs Full Replay | 75%, 프레임 24 | 0/8 | 41 | 10.854303 |
| CPU Native vs Full Replay | 25%, 프레임 2 | 6/6 | 0 | 0 |
| CPU Native vs Full Replay | 50%, 프레임 4 | 4/4 | 0 | 0 |
| CPU Native vs Full Replay | 75%, 프레임 6 | 2/2 | 0 | 0 |

CPU에서는 재생 구간을 포함한 각 실행의 9프레임 전체에서 모든 비교 필드가 정확히 일치했다. MPS에서는 구간 분리 이전인 프레임 1부터 로짓과 기억 특징 차이가 발생했다. 전환 이후 최대 301픽셀 차이는 전체 540×960 이미지의 약 0.0581%다. 이는 객체 IoU나 GT 정확도 차이가 아니다.

MPS Native 반복 대조에서는 두 실행 모두 추적 구간을 나누지 않았음에도 마스크가 9프레임 중 2프레임만 정확히 일치했다. 최대 다른 픽셀은 245개이며, 다른 필드의 최초 차이는 프레임 1이다. 따라서 MPS 차이를 Full Replay 구간 재시작의 고유 오류로 해석할 수 없다. 특정 MPS 연산의 비결정성이 원인인지까지는 확인하지 않았다.

현재 확인된 결론은 **CPU의 짧은 실험에서는 두 경로가 정확히 일치하지만, 현재 MPS에서는 독립 반복 실행의 정확한 일치가 성립하지 않는다**는 것이다. 기존 표의 Native/Full Replay 동점은 여전히 동일 결과 재사용에 따른 것이므로 독립 재현성 근거로 제시하면 안 된다. CUDA BF16과 전체 데이터셋에 대한 독립 재실행은 이 테스트 범위에 포함되지 않는다.

추가 실행 명령:

```bash
test11/.venv/bin/python -u test12/check_native_replay.py --device cpu --frames 9 --min-pre-frames 2 > test12/run_cpu.log 2>&1
test11/.venv/bin/python -u test12/check_native_replay.py --device mps --frames 9 --control-only > test12/control_mps.log 2>&1
```

`results_cpu.json`, `control_mps.json`은 추가 원시 결과다. `summary.json`은 세 실험의 집계다. MPS 실험 두 개의 종료 코드 1은 검출된 불일치를 의미하며, CPU 실험의 종료 코드는 0이다.

## 여러 영상의 구조 검증으로 확장

`compare_multiple_videos.py`는 로컬 DAVIS `bmx-bumps`, `camel`, `breakdance`의 앞 33프레임과 첫 프레임 GT 객체 마스크를 사용한다. 각 영상은 Native 2회, 독립 연속 Full Replay 1회, 25/50/75% 분리 Full Replay 각 1회로 검사한다. 실제 객체는 영상마다 1개씩 선택한다. 같은 모델 가중치를 공유하지만 추론 상태와 입력 reader는 실행마다 새로 만든다.

비교에는 원본 해상도 로짓과 GT J/F/J&F가 추가된다. 전환 지점에서는 입력 reader를 제외한 `inference_state` 전체를 재귀 비교한다. 새 iterator로 이어갈 때 추가 preflight가 상태를 변경하는지 확인하며, 단일 프레임 추론 호출·기억 인코딩 요청 플래그와 모델 가중치의 실행 전후 해시도 검사한다. `_run_memory_encoder` 래퍼 로그는 초기 프롬프트 경로만 포함하며, 후속 프레임의 실제 기억 인코더 모듈 전체 호출을 직접 계측한 로그는 아니다. 후속 기억 텐서는 모든 프레임에서 비교했다. 조건부 동등성의 논증은 `STRUCTURAL_ARGUMENT_KO.md`에 있다.

```bash
test11/.venv/bin/python -u test12/compare_multiple_videos.py --device mps > test12/extended_mps.log 2>&1
test11/.venv/bin/python -u test12/compare_multiple_videos.py --device cpu --deterministic > test12/extended_cpu.log 2>&1
test11/.venv/bin/python test12/summarize_multiple_videos.py
```

큰 추론 상태를 검사하므로 두 장치 작업을 동시에 실행하면 이 Mac에서 메모리 압력이 높아진다. 실제 실행에서는 첫 영상의 MPS Native 추적 일부가 최초 CPU 작업과 겹쳤다. CPU 작업은 첫 경로 완료 전에 종료했으며, GPU 검증 후 CPU 전체를 단독으로 다시 실행한다. 추가 MPS 대조 두 개는 CPU 작업 없이 실행했다. 계측 시간은 성능 비교에 사용하지 않는다.

추가 MPS Native 반복 확인:

```bash
test11/.venv/bin/python -u test12/probe_mps_determinism.py > test12/mps_default_probe.log 2>&1
test11/.venv/bin/python -u test12/probe_mps_determinism.py --deterministic > test12/mps_deterministic_probe.log 2>&1
```

기본 설정의 9프레임 반복은 마스크 2/9프레임만 정확히 같았고, 결정적 연산 옵션을 적용한 반복은 1/9프레임만 정확히 같았다. 두 경우 모두 모든 출력 필드의 정확한 일치를 확보하지 못했다. 이 관측으로 특정 MPS 연산의 원인을 확정하지는 않는다.

원시 결과는 `extended/cpu/*.json`, `extended/mps/*.json`이다. `extended/frame_comparisons.csv`에는 모든 프레임의 마스크 IoU, 다른 픽셀 수, 원본 해상도 로짓 최대/평균 오차와 RMSE, GT J/F/J&F 차이가 있다. `extended/condition_summary.csv`는 전환 조건별 집계, `extended/video_audit_summary.csv`는 전체 상태·재시작·호출 순서 검증이다. 확장 실험 패키지 목록은 `requirements-lock.txt`를 참조한다.

최종 결과와 논문용 표현은 `MULTIVIDEO_REPORT_KO.md`에 정리했다. 두 장치의 입력·가중치·소스 일치와 결과 파일 일관성 검사는 통과했다. CPU는 출력 비교 495/495, 전체 상태 비교 45/45가 정확히 일치했다. 이 합계에는 Native 반복 대조 99개 출력/9개 상태가 포함된다. Full Replay만 분리하면 출력 396/396, 전체 상태 36/36이 정확히 일치했다. MPS는 Full Replay 출력 16/396, 전체 상태 0/36만 정확히 같았고 Native 반복도 4/99만 정확히 같았다.

결과 검증 및 보고서 재생성:

```bash
test11/.venv/bin/python test12/validate_multiple_videos.py
test11/.venv/bin/python test12/summarize_multiple_videos.py
test11/.venv/bin/python test12/write_multivideo_report.py
```
