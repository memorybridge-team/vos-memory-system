# 협업·연구 운영 요구사항 초안

> 작성일: 2026-09-19 KST. 이 문서는 사용자가 검토·삭제·보완할 **요구사항 초안**이다. 연구의 확인된 사실은 `PROJECT_CONTEXT.md`, 구체적 실행 순서는 `docs/experimental_plan.md`를 기준으로 한다.

## 1. 연구의 현재 방향

- 주 연구 쌍은 SAM 2.1 **Small → Base+**다. `Base+`는 공식 모델명이다.
- MOSEv2/LVOS v2 train-fit·dev로 translator를 학습·선택하고 official validation은 sealed
  in-domain final로 사용한다. VOST는 primary, PUMaVOS와 M³-VOS는 complementary external zero-shot으로 사용하며 DAVIS는
  현재 연구 범위에서 제외한다. 외부 benchmark는 학습·통계·선택에 사용하지 않는다.
- 제안 방식은 nonlinear state/memory translator다. Linear/Ridge는 새 제안 방식이 아니라 과거의 단순 비교 관찰로만 남긴다.
- 목표는 state tensor가 닮아 보이는지가 아니라, 전환 뒤 Base+가 객체를 정확하고 안정적으로 이어 추적하면서 full replay 비용을 줄이는지 확인하는 것이다.
- 1개월은 제출 일정이며, 필요한 데이터·반복·비교군을 임의로 축소하는 근거가 아니다.

## 2. 연구 품질의 비협상 조건

- source/target은 같은 video, frame, prompt timeline을 사용한다. handoff 시점에 미래 정답 mask를 새 입력으로 주지 않는다.
- 데이터 분할은 영상 단위다. test로 구조나 하이퍼파라미터를 고르지 않는다.
- 모든 실행에는 코드 commit, SAM 2 upstream SHA, checkpoint hash, dataset/split, seed, config, 실행 명령, 로그와 결과 경로를 남긴다.
- J&F만 보지 않고 switch 직후 성능, 재등장 회복, 부재 중 false positive, ID continuity, handoff latency, replay량, VRAM, state bytes를 함께 기록한다.
- 실패 사례·미완료 데이터셋·편향된 subset은 숨기지 않고 범위와 한계로 밝힌다.

## 3. 비교군 원칙

- 반드시 같은 `(video, object, switch)` manifest에서 Source-only, Base+-native/Full Replay, Direct State Copy, Original-Prompt(s), Last-Visible Source Mask, Original+Last-Visible, Original+Replay-k를 비교한다.
- Last-Mask, Recent Replay-k, Reset은 저비용 진단군이다. 제안 방식의 유일한 경쟁군으로 과장하지 않는다.
- Last-Visible과 Replay-k는 모든 비디오에서 똑같이 강한 방법이라고 가정하지 않는다. 최근 관측·재등장 여부 같은 사건 조건별로 결과를 나눈다.
- Base+ same-checkpoint export→inject는 state 조립 정확성 검사이지, 제안 translator의 성능 경쟁군이 아니다.

## 4. 실행과 비용 운영

- RunPod GPU 실행은 재개 가능한 shard/job으로 구성하고, 로그·checkpoint·manifest·결과를 persistent/network volume에 보관한다.
- 장시간 job이 실행 중이고 다음 판단이 필요 없으면 작업을 기다리지 않는다. 사용자는 RunPod의 GPU utilization, CPU load, process/log 마지막 갱신 시각으로 종료 여부를 확인할 수 있어야 한다.
- 대규모 실험을 예산만으로 축소하지 않는다. 다만 유휴 Pod은 비용을 막기 위해 중지할 수 있고, 재개 절차와 저장 위치를 기록한다.
- 실험 결과는 보고서, 이미지/갤러리, 실행 명령을 남겨 사용자가 IDE·GitHub Pages에서 직접 확인할 수 있게 한다.

## 5. GitHub 협업 운영

1. GitHub Project의 `Tasks`에서 이번 작업을 고르고, 완료 기준과 산출물이 불명확하면 먼저 정한다.
2. 코드·실험·문서처럼 추적해야 하는 일은 `vos-memory-translator-nonlinear`의 Issue로 만든 뒤 Project 카드와 연결한다. 회의·마일스톤만 draft 카드로 둔다.
3. Issue에 목적, 입력 데이터, 완료 기준, 실행 명령 또는 산출물 위치를 적는다.
4. 작업 중에는 Project 상태를 `In Progress`로 바꾸고, 코드·문서·실험 로그를 남긴다.
5. 변경은 작은 단위로 commit하고 push한다. 검토가 필요한 변경은 PR을 열어 Issue를 연결한다.
6. 결과, 실패 조건, 재현 명령, 다음 판단을 Issue에 기록한다.
7. 완료 기준을 실제로 충족하고 링크가 남았을 때만 Issue를 닫고 Project 상태를 `Done`으로 바꾼다. commit만 하고 Done으로 바꾸지 않는다.

## 6. 보드 사용 원칙

- `Tasks`는 오늘 무엇을 하는지 확인하는 표다. 담당자·상태·시작일·목표일·Issue/PR 연결을 관리한다.
- `Roadmap`은 같은 카드들을 시간 순서로 보는 일정도다. 의존 작업이 밀리거나 검증 결과가 바뀔 때만 날짜를 수정한다.
- Draft item은 commit/PR과 자동 연결되지 않는다. 실제 구현을 시작하는 card는 Issue로 승격한다.
- 저장소와 보드의 모델 쌍, 데이터셋, 일정은 같은 표현을 쓴다. 불일치가 발견되면 결과를 변경하기 전에 문서와 보드를 함께 정정한다.

## 7. 문서·소통 방식

- 기본 언어는 한국어이며 모델명·지표·코드 모듈은 통용되는 영문도 함께 쓴다.
- 설명은 비연구자도 따라갈 수 있도록 결론부터 짧고 분명하게 쓴다. 확인된 사실, 가설, 제한적 pilot 결과를 구분한다.
- 사용자가 일간 보고서를 요청하면 `완료 / 다음 작업 / Blocker / 다음 산출물과 ETA / 링크` 형식을 사용하고, 연구를 이해하기 위한 마일스톤 학습도 포함한다.
- 사용자는 2026-09-20부터 핵심 연구 산출물을 Notion Documents의 `모델API&실험`에 자동 업로드하도록 지속 권한을 부여했다. 핵심 산출물은 범위/계약 동결, 구현 gate 보고서, 학습·평가 결과, 의사결정 문서, 논문 표·그림처럼 후속 판단의 기준이 되는 결과다. 단순 로그·중간 cache·반복 실행물은 자동 업로드하지 않고 Git/원격 저장소 링크로 연결한다. 검증된 로컬 스냅샷은 `PROJECT_CONTEXT.md`에도 남긴다.

## 8. 모델 선택과 중단 기준

- 사용자가 매 요청에서 알려준 현재 모델을 해당 작업의 기준으로 사용한다. UI에서 과거 각 프롬프트의 선택 모델을 신뢰성 있게 독립 확인할 수 없다고 가정하며, 알 수 없는 모델을 단정하지 않는다.
- 단순 문서 정리, 보드 확인, 로그·파일 점검은 Terra 수준으로 처리한다.
- SAM 2 구현·디버깅, 실험 설계의 정합성, 결과 원인 분석은 Sol High가 적절하다.
- 새로운 방법론의 방향 전환, 여러 증거가 충돌하는 과학적 판단, 제출 직전의 방법·주장 감사는 Astra가 필요한 시점으로 사용자에게 먼저 알린다.
- 높은 모델이 필요한 판단이 아니면 불필요하게 상위 모델을 사용하지 않는다.

## 9. 사용량 안전 운영

- 새 구현, SSH 기반 환경 변경, 장시간 GPU 실험, 대규모 문헌·결과 판단을 시작하기 직전에 5시간·주간 사용량과 reset 시각을 확인한다.
- 시스템은 업무별 정확한 토큰·credit 소비량을 사전 보장하지 않는다. 따라서 정확한 숫자를 추정했다고 주장하지 않고, 모델·입출력·reasoning·도구 수에 따라 사용량이 달라진다는 전제에서 작은 검증 단위와 저장 지점을 사용한다.
- 5시간 여유가 부족하면 새 장시간 작업을 시작하지 않는다. 대신 현재 branch, 실행 명령, 로그 경로, 산출물 경로, 다음 gate를 기록해 다음 세션이 같은 지점에서 재개할 수 있게 한다.
- 긴 GPU job은 실험 ID, manifest, command, PID, 상태 파일, 로그, 결과 디렉터리와 resume 규칙을 남긴 뒤 background job으로 실행한다. 실행 중 독립적인 판단이 없으면 polling으로 대화를 소모하지 않는다.
- 모델을 바꾸어도 Work/Codex의 공유 allowance가 되돌아오지는 않는다. 작업 난도에 맞는 가장 낮은 모델을 우선 사용하고, 상위 모델이 필요한 판단에서는 먼저 중단·요청한다.

## 10. 사용자가 검토할 항목

- 팀원이 Issue를 만들 수 있는 범위와 PR 리뷰·merge의 최종 담당자를 정할지.
- 일간 보고서의 담당자 표기와 공유 채널을 고정할지.
- GitHub Pages에 공개 가능한 실험 결과와 비공개로 남겨야 할 원자료의 경계를 정할지.
- RunPod persistent/network volume의 표준 경로와 checkpoint 보존 기간을 정할지.

## 11. 답변별 운영 규칙

- 매 답변 시작 시 업무 복잡도에 적합한 모델인지 판단한다. 단순 정리·조회는 Terra Medium, 구현·디버깅·실험 설계는 Sol High, 방법론 전환이나 제출 직전 과학적 감사는 Astra를 우선 검토한다.
- 현재 모델이 부적절하면 작업을 진행하지 않고 필요한 모델을 먼저 요청한다.
- 매 최종 답변에는 전체 계획과 현재 위치를 짧게 표시한다.
- 한 프롬프트에 여러 요청이 있으면 각 요청을 체크리스트로 나누고, 최종 답변에서 `완료 / 진행 중 / Blocker / 미착수` 상태를 각각 보고한다. 순서대로 처리하지 못한 항목도 병목과 독립 진행 여부를 구분한다.
- 장시간 GPU job이 실행 중이고 즉시 할 독립 작업이 없으면 불필요하게 기다리지 않는다. 종료 확인 방법과 다음 재개 조건을 알려주고 답변을 마친다.
- 매 답변에는 Project 카드/Issue/branch/증거 상태를 함께 짧게 보고한다. 이 보고는 실제 GitHub 상태와 로컬 산출물을 구분한다.
