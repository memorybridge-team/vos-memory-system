# Tests

`src/vos_memory_inspector/`의 모듈 경계를 따라 구성한 synthetic 회귀 테스트입니다. GPU, 데이터셋, SAM 2 checkout 없이 `pytest`로 실행됩니다.

- `test_preconditioning.py`: canonical transform round-trip, anchor 보존, seed 재현성, split·pair type 검증, pair bank, split manifest
- `test_checkpoint_audit.py`, `test_data_acquisition.py`, `test_artifacts.py`: audit 경로 정책, archive checksum 검증, SAM 2 frame index 순서
- 나머지: 상위 프로젝트에서 가져온 상태 추출·주입, manifest, 평가 도구
