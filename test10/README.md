# test10 — Small → Base+ 상태 전환 실험

Small이 frame `t`까지 처리한 상태를 Base+에 전달해 `t+1`부터 이어 처리하는 방법을 비교한다. 결과 패키지에는 평가 harness, 선택 manifest, case별 점수 JSON, 요약 통계, 실행·비용·provenance 기록이 포함되어 있다.

## 결과 요약

- 4시간 누적 예산 중 **157.5분** 사용. Smoke 32 cases와 full schedule 116 cases, 총 **148 cases** 완료.
- MOSEv2 77 cases / 77 videos, LVOSv2 41 / 40, DAVIS 2017 30 / 30. 전체 147개 고유 영상이다. 객체별 독립 실행이며 다객체 동시 추적 결과로 해석하지 않는다.
- 13개 품질 방법 모두와 self-injection gate가 148 cases에서 완료됐고, 실패 attempt는 없다. Self-injection은 148/148 통과(logit max error 0, binary IoU 1.0).
- 기존 fit shard와 translator checkpoint는 재사용했고 새 학습은 하지 않았다. 과거 test9 점수 row는 provenance가 불충분해 재사용 0건으로 기록했다.

J&F (point, 데이터셋별 영상 평균):

| Dataset | Direct Copy | Affine `Wx+b` | Residual MLP | Transformer | Last-mask | Anchor Replay-16 |
|---|---:|---:|---:|---:|---:|---:|
| MOSEv2 | 31.42 | 64.85 | 60.12 | 64.90 | 60.69 | 65.98 |
| LVOSv2 | 3.64 | 94.26 | 95.24 | 95.21 | 95.01 | 86.72 |
| DAVIS 2017* | 10.42 | 90.02 | 90.08 | 89.97 | 90.24 | 84.61 |

Affine은 Direct Copy보다 크게 높았다. Nonlinear의 affine 대비 paired 차이는 MOSEv2에서 MLP −4.73 [−9.36, −1.05], Transformer +0.05 [−2.93, +2.71]; LVOSv2에서 MLP +0.99 [−0.07, +3.04], Transformer +0.96 [−0.12, +3.03]; DAVIS에서 각각 +0.06 [−0.97, +0.99], −0.05 [−0.20, +0.11] point였다. 따라서 nonlinear가 데이터셋 전반에서 affine보다 낫다는 근거는 확인되지 않았다.

Component ablation에서는 spatial-only affine이 full affine에 가까운 점수를 보였고, pointer-only affine은 Direct Copy와 비슷했다. 이번 조건에서는 성능 개선이 주로 spatial memory 변환에서 나왔다. Affine과 last-mask/replay 간 paired CI는 대체로 0을 포함해 일관된 우위가 확인되지 않았다.

동일한 32-case smoke 비용 비교에서 중앙 handoff 시간은 affine 약 7.4ms, Anchor Replay-16 약 560ms였다. 첫 suffix 출력까지는 각각 약 57ms, 604ms였다. 이는 warm-model 고정 smoke 계측이며 모델 load 및 video 초기화 시간은 별도 기록이다.

* DAVIS validation 영상의 과거 모델 선택 사용 여부를 확인하지 못해 탐색적 결과로 표시한다. 전체 지표·95% bootstrap CI·paired 비교·비용 표는 [summary.md](runs/run01/summary.md)와 [summary.json](runs/run01/summary.json)을 참조한다.

## 포함 파일과 큰 산출물

`runs/run01/artifacts/*.json`에는 case별 점수와 self-injection 결과가 있고 `.sha.json` sidecar는 content-addressed artifact의 hash를 보존한다. 크기가 큰 `.pt` 예측/state blob 1,924개(약 40.44 GiB)는 GitHub 저장소에 올리지 않았다. 따라서 점수·통계는 검토할 수 있지만, 이 공개 패키지만으로 binary mask/state를 복원하거나 해당 cache에서 `--resume`할 수는 없다.

선택 manifest와 provenance의 로컬 절대 경로는 공개를 위해 `${WORKSPACE_ROOT}`로 치환했다. 입력 파일 SHA-256, checkpoint SHA-256, 모델·소프트웨어 정보는 유지했다. 원본 데이터, SAM 2 checkpoint, test9 translator 코드/checkpoint 및 600 fit shard는 별도 자산이며 이 폴더에 포함되지 않았다.

## 재현 관련 주의

CPU 계약 테스트 22개가 통과했다. 실험은 SAM 2 + CUDA 환경에서 실행됐으며, 선택적 `_C` post-processing extension을 불러오지 못해 fill-holes 후처리를 건너뛴다는 upstream 경고가 기록됐다. 추론과 gate는 완료됐지만, 이 동작은 재현 환경에서 확인해야 한다.

Harness는 실행 시 sibling `test9/`, `vos-data/`, `vos-checkpoints/`, `sam2/`와 맞는 Python/CUDA 환경을 기대한다. 현재 GitHub 저장소 clone만으로는 이 실험을 독립 재실행할 수 없다. 따라서 `test9` 패키지가 없는 clone에서는 해당 패키지를 import하는 affine/model adapter 테스트도 실행되지 않는다(현재 repository-only 확인에서 `mvp_common` 부재로 실패). 원 실험 workspace에서는 CPU 계약 테스트 22개가 통과했다. 필요한 데이터·checkpoint 권리와 경로를 준비한 뒤 실행한다. 이 실험 실행의 source 경로/hash와 설정은 [provenance.json](runs/run01/provenance.json)에 기록되어 있다.

기본 명령(위 외부 자산이 준비되어 있고, workspace root에서 실행):

```sh
python -m unittest discover -s test10/tests -v
python test10/run.py --stage audit --run-dir test10/runs/run01
python test10/run.py --stage smoke --run-dir test10/runs/run01 --resume --budget-hours 4
python test10/run.py --stage full --run-dir test10/runs/run01 --resume --budget-hours 4
```
