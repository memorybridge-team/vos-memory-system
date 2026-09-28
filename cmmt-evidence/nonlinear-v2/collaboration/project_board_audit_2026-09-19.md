# GitHub Project #2 task 감사·수정 기록

기준일: 2026-09-19 KST. Project의 기존 20개 연구 카드는 제목뿐 아니라 본문, 담당자, 상태, 시작일·종료일을 전부 확인했다. 본문이 비어 있거나 완료 기준이 약한 카드는 연구 범위, 실행 증거, 재현성, 통계·누수 검토 기준을 보완했다.

## 결론

연구의 뼈대는 **01 Scope → 20 Submission**의 20단계로 충분하다. 번호는 실행·관리 순서를 뜻하며, 장시간 학습과 정기 회의처럼 일부 단계는 서로 겹쳐 진행할 수 있다. `Infra`, canonical Issue, PR은 별도 연구 단계가 아니라 기존 단계의 실행 환경·증거이므로 21번 이후 단계로 세지 않는다.

Tasks 기본 보기는 제목 오름차순으로 저장했다. 따라서 보드를 열면 `01`부터 `20`까지 순서대로 보인다.

## 확정한 20단계

| # | Task | 핵심 완료 기준 |
|---:|---|---|
| 01 | Scope 고정 | Small→Base+, MOSE/LVOS in-domain과 VOST/PUMaVOS/M³-VOS external 역할, nonlinear 범위, baseline, 성공·실패 기준과 target-native 필요성 판정 규칙을 동결. DAVIS는 현재 연구 범위에서 제외. hard-event 실험 실행은 후속 pilot/evaluation task |
| 02 | Small/Base+ memory I/O 계약 | component별 shape·dtype·의미·정렬·복사/번역/재생성 정책, 예시 dump, runtime inventory와 State Assembly Map 동결 |
| 03 | 데이터·난이도·baseline·metric 고정 | switch manifest, 누수 금지, video-clustered CI, 실패·제외 규칙 |
| 04 | 논문 질문·기여·개요 | 반증 가능한 RQ, claim–evidence 대응, 과장 없는 novelty 범위 |
| 05 | 멘토 검토 agenda | 결과·실패·결정 질문·후속 action 링크를 회의마다 누적 |
| 06 | memory export/self-injection/target injection | export·materialize·inject 구현, Base+ same-checkpoint 연속 실행 일치, Small→Base+ target injection, 다객체·late prompt·부재/재등장·prompt correction·복수 영상/switch, field trace와 hash |
| 07 | paired-state 수집 | MOSE/LVOS fit/dev의 future-GT-free `(b_T,a_T)` pair, video-level split, checksum manifest, resumable shard, storage capacity 보고 |
| 08 | evaluator·필수 baseline 구현 | 동일 manifest·공식 evaluator·미래 GT 금지·결과 일치 test |
| 09 | nonlinear 후보 구현·비교 | residual/gated/slot-context 후보, state-only primary, GT-free distillation·supervised rollout ablation, dev-only selection |
| 10 | single-video overfit·pilot | overfit 통과 수치, held-out smoke, deterministic reload |
| 11 | 전체 학습·반복 실험 | seed·checkpoint 선택·중단/재개·실패 run 보존 |
| 12 | 후보·프로토콜 중간 검토 | go/no-go rubric, 결정 근거, 남은 실행 matrix |
| 13 | 전체 baseline·조건 분석 | 어려운 사건 strata, retry 규칙, CI, latency/VRAM/state bytes |
| 14 | 방법·실험·예비 결과 통합 | notation과 claim consistency, section별 근거 링크 |
| 15 | 결과 동결 검토 | 제외 사례와 실패 run을 포함한 freeze manifest |
| 16 | 표·그림·전체 초안 | 원본 결과 JSON과 생성 명령 연결, 수동 전사 금지 |
| 17 | 내부 검토·인용·수정 | claim–evidence, 인용, 통계, 누수, 한계 checklist |
| 18 | 재현성·release 검증 | lockfile, CUDA/PyTorch/SAM revision, checkpoint/data hash, clean run, artifact sync |
| 19 | 제출 파일·모의 제출 | 공식 형식·마감 시간대·등록/결제·업로드 dry run 확인 |
| 20 | IEIE 2026 제출 | 접수번호, 최종 파일 hash, 원고 보관 위치 기록 |

## 기존 20개에서 실제로 보완한 내용

1. Scope Issue #1의 `Tiny → Base+`를 `Small → Base+`로 고쳤고 target-native 필요성, same-checkpoint injection, 필수 baseline, 통계·누수 기준을 추가했다.
2. Dataset/benchmark 카드의 같은 오기를 고치고 switch selection rule, 사건별 표본 수, video-clustered confidence interval을 추가했다.
3. 설명과 완료 기준이 약했던 카드에는 실행 matrix, 실패·재시도 규칙, checkpoint 선택, clean-run 재현, 표·그림 원본 연결, 제출 증거를 보완했다.
4. 원래 담당자와 완료 상태는 연구 근거 없이 임의로 바꾸지 않았다.

## 20단계 밖 항목의 처리 원칙

- `[Infra] Freeze RunPod environment, storage and artifact sync`의 요구사항은 독립 연구 단계가 아니다. CUDA/PyTorch/SAM revision과 checkpoint hash는 06·18, Network Volume의 resumable shard와 checksum은 07, 완료 marker·로그·artifact sync는 11·18의 완료 기준으로 흡수한다.
- 02와 06을 같은 완료 조건으로 묶지 않는다. Issue #2는 I/O 계약과 실제 Small/Base+ inventory·paired dump·State Assembly Map 검토를 추적한다. export/self-injection/target injection의 구현 및 edge-case continuation 검증은 별도 06 task에서 추적한다.
- PR #3은 연구 단계가 아니라 Issue #2의 코드·테스트 증거다. 검토 전 `main`에 직접 반영하지 않는다.
- 예전에 만든 `Map ... select translator I/O` draft는 Issue #2와 중복이어서 보관 처리했다. 별도 Infra 카드와 PR #3 Project 카드도 보관했으며, 원본 PR은 열어 둔 채 Issue #2에서 코드·테스트 증거로 추적한다.

## 현재 위치

- 01은 baseline·공정성 규칙·성공·중단 기준을 `docs/design/01_scope_baselines_success_stop.md` v1.0으로 동결해 Done 기준을 충족했다.
- 02는 정적 contract·validator·실제 Small/Base+ runtime inventory·paired dump·State Assembly Map을 v1.0으로 동결해 Done 기준을 충족했다.
- 03 v1.0의 DAVIS·MOSEv2·LVOS v2 기록은 역사적 milestone으로 보존한다. 현재 활성 protocol은 MOSEv2·LVOS v2 in-domain, VOST primary external, PUMaVOS·M³-VOS complementary external을 사용하며 두 onboarding 완료 전까지 Task 03은 In Progress다.
- 06은 단일/다객체·late prompt·부재/재등장, 전환 전후 correction, 반복 handoff의 Base+ same-checkpoint gate와 Small→Base+ Direct Copy target injection을 검증해 Done 기준을 충족했다.
- 2026-09-24 protocol v1.1에서 in-domain과 external zero-shot을 분리하고 VOST를 추가했으므로 03을 다시 `In Progress`로 연다. 기존 v1.0 결과는 취소하지 않는다.
- 다음 순서는 03 PUMaVOS·M³-VOS onboarding → 07 paired-state 수집 → 08 evaluator/baseline → 09~11 nonlinear 학습이다.

## 2026-09-24 protocol v1.1 보드 수정

- **03 Benchmark:** fit/dev·sealed final·external 역할, VOST license/download/checksum, manifest·loader·`J/J_last`, external access ledger를 완료 조건에 추가한다.
- **03 Benchmark:** PUMaVOS 공식 archive download/checksum·dense inventory·23/24 출처 표기 불일치 해소·first-nonempty prompt·fixed manifest·J/F/J&F contract를 추가한다.
- **03 Benchmark:** M³-VOS access/delivery revision·full/core inventory·void-aware prompt loader·`J/J_last/J_cc` contract와 boundary F integrity gate를 추가한다.
- **07 Paired state:** MOSEv2/LVOS v2 fit/dev만 training shard에 넣고 future dataset GT를 primary pair loss에 쓰지 않는다. DAVIS/VOST/PUMaVOS/M³-VOS는 training paired-state source에서 제외한다.
- **08 Baselines:** Moment-Matched 통계는 fit shard로만 계산하고 VOST `J/J_last`, PUMaVOS local J/F/J&F, M³-VOS `J/J_last/J_cc`와 조건부 보조 F/J&F, in-domain/external 결과 분리를 구현한다.
- **09–11 Training:** state-only를 primary로 하고 GT-free distillation·supervised rollout을 ablation으로 분리한다. 모든 architecture/loss/checkpoint 선택은 MOSEv2/LVOS v2 dev에서 끝내고 config freeze 후 official validation과 external benchmark를 연다.
- **13 Evaluation:** LVOS local detailed final, MOSE Codabench sealed final, VOST primary external, PUMaVOS partial/unusual-mask stress, M³-VOS material phase-transition stress를 다른 표로 보고한다. DAVIS 결과는 현재 연구의 표·주장에 포함하지 않는다.

Project의 01→20 뼈대와 번호는 바꾸지 않는다. 이번 결정은 새 task를 추가하는 것이 아니라
03·07·08·09–11·13의 데이터 노출 계약과 완료 기준을 강화한 것이다.
