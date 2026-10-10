# test12: SAM2 Base+ Native와 Full Replay의 독립 검증

2026-10-10에 Apple M2 Mac에서 실행한 실제 체크포인트 실험이다. 같은 Base+ 가중치로 Native와 Full Replay를 독립 실행하고, 프레임별 출력과 전체 추론 상태를 비교했다.

## 결과

DAVIS `bmx-bumps`, `camel`, `breakdance`의 앞 33프레임을 사용했다. 영상마다 객체 하나를 선택하고 첫 프레임 GT 마스크를 프롬프트로 제공했다. 전환 프레임은 8/16/24이며, 장치마다 영상당 Native 2회와 Full Replay 4회(연속 1회, 전환별 분리 3회)를 실행했다. 총 36회, 1,188프레임 추론이다.

| 비교 | CPU FP32, 결정적 연산 설정 | MPS FP32, 기본 설정 |
| --- | ---: | ---: |
| Full Replay의 모든 출력 필드 정확히 일치 | 396/396 | 16/396 |
| Full Replay의 전체 상태 정확히 일치 | 36/36 | 0/36 |
| Native 반복의 모든 출력 필드 정확히 일치 | 99/99 | 4/99 |
| 분리 재시작의 추가 preflight가 상태를 보존 | 9/9 | 9/9 |

CPU에서는 검사한 범위에서 모든 값이 정확히 일치했다. MPS에서는 Native 반복부터 차이가 발생했다. 따라서 실행마다 오차가 반드시 생기는 것은 아니며, MPS 차이를 Full Replay 전환 자체의 오류로 단정할 수 없다. 동일한 상태 전이와 결정성을 전제로 하는 알고리즘 동등성은 설명할 수 있지만, 모든 영상·장치에서의 무조건적인 수치 일치를 증명한 것은 아니다.

기존 benchmark 표의 Native/Full Replay 동점은 같은 결과 재사용에 따른 것이다. 이번 실험은 별도 상태와 입력 reader를 생성했고, Native 예측을 Replay 입력으로 사용하지 않았다.

## 자료 안내

- [최종 분석과 논문용 결론](MULTIVIDEO_REPORT_KO.md)
- [상태 전이와 조건부 동등성 논증](STRUCTURAL_ARGUMENT_KO.md)
- [초기 bedroom 진단 실험](INITIAL_EXPERIMENT_KO.md)
- [프레임별 비교 CSV](extended/frame_comparisons.csv): 990행, 로짓 오차·마스크 IoU·GT J/F/J&F 차이.
- [전환 조건별 CSV](extended/condition_summary.csv), [영상별 상태 검사 CSV](extended/video_audit_summary.csv)
- 원시 결과: [CPU](extended/cpu), [MPS](extended/mps), 각 장치의 `environment.json`과 영상별 JSON.
- [최종 결과 검증](extended/validation.json), [실험 당시 설정](extended/configuration_manifest.json)
- [소스 출처와 해시](provenance/source_manifest.json): 실험 당시 원본 스크립트, benchmark 실행 모듈과 핵심 SAM2 소스.
- `*.log`: 당시 실행 로그. `results_*.json`, `control_mps.json`, `mps_*_probe.json`: 초기 및 추가 대조 결과.

IoU는 두 예측 사이의 일치도다. GT 정확도와 구분해서 해석해야 한다. CSV의 전환별 구간 길이는 다르며, 보고서의 25/50/75% 평균은 세 전환에 같은 비중을 준다.

## 결과 검증과 집계 재생성

저장소 루트에서 실행한다. 아래 명령에는 모델, 데이터셋, PyTorch가 필요하지 않다.

```bash
python3 test12/verify_package.py
python3 test12/validate_multiple_videos.py
python3 test12/summarize_multiple_videos.py
python3 test12/write_multivideo_report.py
```

`verify_package.py`는 원시 결과 해시, 실험 소스 해시와 fixture 출처를 검사한다. `validate_multiple_videos.py`는 프레임·경로·상태의 커버리지와 입력·가중치·비교값의 일관성을 검사한다. 수치 불일치는 MPS에서 관측한 실험 결과이며 파일 검증 실패를 의미하지 않는다.

## 추론 재현

Python 3.12.14 환경에서 실행했다. benchmark 모듈은 `runtime/`에 포함했으므로 별도 benchmark checkout과 `test11` 폴더가 필요하지 않다. 가중치, DAVIS RGB/GT, SAM2 전체 소스와 가상환경은 포함하지 않는다.

SAM2 소스는 `facebookresearch/sam2`의 커밋 `2b90b9f5ceec907a1c18123530e92e794ad901a4`를 사용했다. `SAM2_ROOT`는 `sam2/`, `checkpoints/`, `data/`를 포함하는 해당 checkout의 절대 경로다. 결과 JSON과 `provenance/source_manifest.json`에 체크포인트·입력·소스 해시를 기록했다.

```bash
git clone https://github.com/facebookresearch/sam2.git /path/to/sam2
git -C /path/to/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
export SAM2_ROOT=/path/to/sam2
python3.12 -m venv test12/.venv
test12/.venv/bin/python -m pip install -r test12/requirements-lock.txt
SAM2_BUILD_CUDA=0 test12/.venv/bin/python -m pip install --no-deps --no-build-isolation -e "$SAM2_ROOT"
```

입력 파일을 다음 위치에 준비한다.

```text
$SAM2_ROOT/checkpoints/sam2.1_hiera_base_plus.pt
$SAM2_ROOT/data/DAVIS/JPEGImages/480p/{bmx-bumps,camel,breakdance}/*.jpg
$SAM2_ROOT/data/DAVIS/Annotations/480p/{bmx-bumps,camel,breakdance}/*.png
$SAM2_ROOT/notebooks/videos/bedroom/*.jpg  # 초기 진단 실험에만 사용
```

Base+ 체크포인트 SHA256: `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`. 주 실험은 RGB와 GT의 앞 33프레임을 사용한다.

```bash
test12/.venv/bin/python -u test12/compare_multiple_videos.py --device cpu --deterministic
test12/.venv/bin/python -u test12/compare_multiple_videos.py --device mps
test12/.venv/bin/python -u test12/probe_mps_determinism.py
test12/.venv/bin/python -u test12/probe_mps_determinism.py --deterministic
test12/.venv/bin/python -u test12/check_native_replay.py --device cpu --frames 9 --min-pre-frames 2
```

장치 작업은 순서대로 실행한다. 다영상 재실행 결과는 `extended/rerun-cpu/`, `extended/rerun-mps/`에, 진단 결과는 `reruns/`에 저장한다. 게시된 `extended/cpu/`, `extended/mps/`는 보존한다. 집계 스크립트는 게시된 두 장치 결과만 읽는다. 초기 비교 스크립트의 종료 코드 1은 출력 불일치를 뜻한다.

게시를 위해 경로 설정과 결과 저장 위치만 조정했다. 실험 당시 스크립트는 `provenance/original_scripts/`에 원본 그대로 보존했다. 게시용 경로 조정 후 전체 추론 36회를 다시 실행하지는 않았다.

## 해석 범위

3개 영상의 처음 33프레임과 영상별 객체 하나를 검사했다. 전체 DAVIS/PUMaVOS/VOST, 동시 다객체, 추가 프롬프트, CUDA BF16, Small→Base+ 전이는 검사하지 않았다. 실행 시간에는 상세 계측이 포함되어 성능 비교에 쓸 수 없다. `_run_memory_encoder` 로그는 최초 프롬프트 래퍼 경로만 계측하며 후속 기억 인코더 모듈의 전체 호출 로그가 아니다. 후속 기억 텐서 값과 인코딩 요청 플래그는 검사했다.

기본 MPS와 결정적 연산 옵션의 짧은 대조 모두 완전 일치를 확보하지 못했다. 오차 원인과 오차 분포의 통계적 동등성은 확인하지 않았다. 전체 텐서 원본 대신 필드별 해시·수치 비교값·메타데이터를 보존했다. SAM2 소스 발췌의 라이선스는 [Apache-2.0](provenance/sam2/LICENSE)이다.
