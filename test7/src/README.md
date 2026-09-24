# Source code

현재 SAM 2 memory inspection 및 compatibility 구현은 `src/vos_memory_inspector/`에 있습니다. 이후 translator 구현을 추가할 때에도 state schema/validator, export-inject round-trip, translator, loss/trainer, evaluator integration의 경계를 분리하는 것을 원칙으로 합니다.
