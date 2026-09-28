# 기존 저장소에서 v2로 옮긴 범위

기준일: 2026-09-18 KST. 원본 저장소: `memorybridge-team/vos-memory-translator-nonlinear`의 당시 `main` 및 로컬의 미커밋 연구 문서.

## 이관한 것

- `src/vos_memory_inspector/`: 실제 SAM 2 state 추출·주입, canonical schema, cache, DAVIS 평가, MLP 코드. 일부 Ridge/Linear 경로는 과거 결과 재현과 회귀 검사용으로 남겨 두었다.
- `tests/`: 기존 동작 검증. 새 Base+/MOSEv2/LVOS v2 검사는 이후 추가한다.
- `scripts/`: paired-state 수집, nonlinear 학습·평가, 기존 baseline, 결과 집계에 쓰는 명령만 선별했다.
- `SAM2_UPSTREAM_COMMIT`, `pyproject.toml`, `.gitignore`, DAVIS manifest 정책.
- 구조 이해에 필요한 조립 지도와 핵심 선행 연구 설계 문서. 기존 Tiny→Large 결과 중 규모가 작은 MLP 보고서와 rare-event 요약은 `reports/legacy/`에 옮겼다.

## 새로 작성하거나 고친 것

- `README.md`, `PROJECT_CONTEXT.md`, `docs/experimental_plan.md`: Tiny→Base+, 세 데이터셋, 4주 논문 일정과 nonlinear 범위로 재작성.
- `scripts/runpod_bootstrap.sh`: 새 기본 target을 Base+로 변경.
- 새로운 checkpoint 기반 Base+ 결과, MOSEv2/LVOS v2 로더와 객체별 anchor baseline은 구현·검증 전이다. 존재하지 않는 실험 결과를 옮겨 쓰지 않는다.

## 이관하지 않은 것

- 과거 Tiny→Large 전용 대량 gallery·이미지, 오래된 날짜별 보고서 대부분, 과거 학습용 대형 tensor cache, 원본 데이터·checkpoint, RunPod SSH 정보.
- `run_tiny_large_probe.py`와 `runpod_direct_smoke.sh`처럼 모델 쌍이 고정된 예전 실행 스크립트.
- 과거 Git commit history. 새 저장소의 첫 commit은 새 루트이며 이전 commit을 parent로 가져오지 않는다.

## 작성 기여와 복구

새 저장소의 초기 commit 작성자는 GitHub 계정 `KIMKYUDO`로 설정한다. 이는 **파일 전체를 KIMKYUDO가 처음 작성했다는 뜻이 아니다.** 이전 저장소에는 `KyudoKim`과 `서수빈`의 커밋이 있으며, 후자는 초기 memory probe 등 일부 이관 파일에 기여했다. 과거 저장소의 `git log`와 이 문서가 작성 경위를 설명한다. 원본 Git 이력과 전체 산출물을 복구할 수 있도록 로컬 기존 checkout의 `.git`을 보존한다. 원본 GitHub 저장소를 삭제하면 그 저장소의 Pages URL 및 GitHub 상의 history 접근도 사라진다.

새 GitHub 저장소의 Contributors 표시는 새 commit의 작성 이력에 의해 계산된다. 초기 commit을 단일 GitHub 계정의 검증 가능한 이메일로 작성한 뒤 확인한다. 이후 팀원 기여가 생기면 정당한 작성자를 그대로 기록한다.

## 2026-09-18 이관 실행 결과

- 새 저장소 `main`의 독립된 첫 commit `dc1d667e0a4f42739f1e86d66f2715e06b11f9c2`를 push했다. GitHub Contributors API에서 `KIMKYUDO` 한 명만 확인했다.
- 새 저장소 GitHub Pages를 `main /docs`로 설정했다. 공개 주소 `https://memorybridge-team.github.io/vos-memory-translator-nonlinear-v2/`의 HTTP 200 응답을 확인했다.
- 기존 원격 `main`과 로컬 checkout의 commit이 일치하며, 원격 branch는 `main` 하나이고 tag는 없다. 원본 이력 백업 `.external/archives/vos-memory-translator-nonlinear-before-v2.bundle`을 만들고 `git bundle verify`로 검증했다. 이 bundle은 기존 로컬 작업공간에만 있다.
- **기존 GitHub 저장소 삭제는 미완료**다. GitHub REST 삭제 요청은 `403 Must have admin rights to Repository`로 거절됐다. 브라우저 UI 진입도 자동 승인 검토에서 거절돼 우회하지 않았다. 저장소 관리자 계정으로 기존 저장소의 Settings → General → Danger Zone에서 삭제해야 한다. 그전에는 기존 저장소와 Pages가 남아 있다.
- 기존 로컬 checkout의 `origin`은 해제했다. 코드, 미커밋 변경, Git 이력과 위 bundle은 그대로 보존하며 새 저장소 `v2/`만 새 원격에 연결한다.
