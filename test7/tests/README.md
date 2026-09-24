# Tests

현재 `src/vos_memory_inspector/`의 모듈 경계를 따라 synthetic test를 구성합니다. State injection 단계에서는 동일-checkpoint export→inject round-trip과 target의 과거-frame encoder 호출이 0회인지 확인하는 integration test를 최우선으로 둡니다.
