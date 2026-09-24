# VOST Base+ handoff diagnostic

이 저장소의 재현 실험은 **VOST validation 영상과 LVOS 일부 영상에서 SAM 2.1 Base+ → Base+ 상태 이전이 같은 추론을 재현하는지** 확인합니다. 학습된 Small→Base+ translator의 성능을 검증하는 실험은 아닙니다.

The reproducible experiments check whether a SAM 2.1 Base+ → Base+ state handoff reproduces uninterrupted inference on a small VOST validation subset and selected LVOS clips. They do **not** evaluate a trained Small→Base+ translator.

## 실험 요약

- VOST val의 전체 영상 2개: `555_tear_aluminium_foil` (51 frames), `6922_split_paper` (78 frames); 이미지·마스크 약 36 MB.
- 각 영상 첫 annotated frame의 모든 객체 mask만 prompt로 사용하고, 영상 중간에서 handoff합니다. Target은 switch 다음 frame부터 추론합니다.
- 같은 Base+ checkpoint를 Source와 Target에 사용하고, `DirectCopyTranslator`로 memory state를 그대로 전달합니다. 학습은 하지 않습니다.
- 성공 조건: 주입 도중 과거 영상 backbone 재계산 없음, 이후 logits/masks/J&F가 native 추론과 허용오차 내에서 일치.
- SAM 2 Target state에는 raw RGB frame들이 로드될 수 있지만, 이는 모든 frame을 미리 추론해 memory를 만든다는 뜻은 아닙니다. 이 실험은 handoff 주입 시 과거 frame backbone 호출이 0회인지 계측합니다.
- 현재 기록 결과: 공식 VOST aggregate `J=0.54257`, `J_last=0.62395`; native와 transferred 간 차이는 둘 다 `0`. 두 영상의 사후 handoff 예측 mask와 logits도 일치했습니다.

영상 두 개만 쓴 제한된 same-checkpoint 진단이므로, 데이터셋 전체 성능이나 cross-model/nonlinear translation의 효용으로 일반화하면 안 됩니다. 프로토콜과 결과 해석은 [`plan.md`](plan.md), 실행 기록은 [`reports/vost/identity-20260924-2clips.md`](reports/vost/identity-20260924-2clips.md)에 있습니다.

## LVOS Base+ identity handoff

2026-09-23에 LVOS v2 영상 4개로 실행한 초기 점검과, 이를 포함해 총 14개 영상으로 확장한 실행이 모두 통과했습니다. 아래는 확장 실행의 결과이며, **공식 LVOS validation 점수는 아닙니다**.

- Protocol: `DirectCopyTranslator`; Source와 Target 모두 같은 SAM 2.1 Base+ checkpoint, seed `7`, CPU.
- 각 영상에서 manifest가 지정한 객체별 최초 prompt만 사용했습니다. 40개 frame을 5-frame 간격으로 평가했고, handoff는 고정된 clip index `20` (원본 frame ID `101`)에서 수행했습니다. Target은 그 다음 frame부터 이어서 추론했습니다.
- Prompt frame은 점수 집계에서 제외했습니다. 아래 J&F는 각 영상의 handoff 이후 **점수에 포함된 frame-object 행**들의 평균입니다. 점수 범위/마지막 frame 제외 규칙 등 공식 LVOS 전체 평가 프로토콜은 적용하지 않았습니다.
- 14/14 영상 통과; post-handoff 361개 frame-object 행에서 native 대비 `ΔJ&F = 0` (평균·최솟값·최댓값), binary mask 전부 동일, 최대 logit 오차 `0`. State injection 중 backbone 재계산은 `0`회였습니다.

| LVOS video | Post-handoff scored rows | Native J&F mean | Transferred J&F mean | Mean ΔJ&F |
|---|---:|---:|---:|---:|
| `0tCWPOrc` | 38 | 0.963010 | 0.963010 | 0.000000 |
| `2VegYEbT` | 19 | 0.972971 | 0.972971 | 0.000000 |
| `2urlAsm8` | 19 | 0.945621 | 0.945621 | 0.000000 |
| `8lxxCA5h` | 19 | 0.954523 | 0.954523 | 0.000000 |
| `9HEh93ef` | 95 | 0.920966 | 0.920966 | 0.000000 |
| `K3OUeINk` | 19 | 0.806886 | 0.806886 | 0.000000 |
| `MKnlVo6x` | 19 | 0.976499 | 0.976499 | 0.000000 |
| `ScFTYisJ` | 19 | 0.946887 | 0.946887 | 0.000000 |
| `aFytsETk` | 19 | 0.985051 | 0.985051 | 0.000000 |
| `dtHbJvYy` | 19 | 0.955877 | 0.955877 | 0.000000 |
| `nfcT3owb` | 19 | 0.967570 | 0.967570 | 0.000000 |
| `q1MSEBkh` | 19 | 0.954201 | 0.954201 | 0.000000 |
| `x3nD3QQ9` | 19 | 0.903679 | 0.903679 | 0.000000 |
| `xpI7xRWN` | 19 | 0.540941 | 0.540941 | 0.000000 |

This is a **same-checkpoint identity/implementation diagnostic**, not evidence that a learned translator improves accuracy or that LVOS-wide performance is validated. The result only shows that, on these sampled frames and this CPU execution path, direct state copying continued the same inference exactly. The CUDA offload path was not exercised. The detailed run files are local under `outputs/lvos_base_roundtrip/20260923T140747Z/` and are excluded from Git; LVOS media and model checkpoints are not included in this repository.

## 재현 방법

Python 3.10+, PyTorch가 포함된 SAM 2 실행 환경이 필요합니다. SAM 2 저장소는 별도 checkout이며 이 실험은 commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`와 아래 checkpoint SHA-256을 검사합니다.

```bash
# test7 저장소 루트에서 실행. SAM 2 checkout은 인접한 ../sam2 에 있다고 가정.
python -m pip install -e ".[dev,vost-eval]"
python -m pip install -e ../sam2
bash scripts/setup_vost_evaluator.sh

# VOST 공식 공개 ZIP에서 아래 두 영상의 필요한 entry만 HTTP Range로 받습니다.
python scripts/vost_subset.py \
  --videos 555_tear_aluminium_foil 6922_split_paper

# 원본 데이터 무결성을 먼저 검사한 뒤 실험 실행
python scripts/evaluate_vost_base_roundtrip.py --check-data-only
python scripts/evaluate_vost_base_roundtrip.py \
  --sam2-repo ../sam2 \
  --checkpoint ../sam2/checkpoints/sam2.1_hiera_base_plus.pt \
  --device auto
```

다른 경로를 쓰려면 `--data-root`, `--sam2-repo`, `--checkpoint`, `--evaluator-repo`, `--output-root` 인자를 지정할 수 있습니다. 새 실행은 기존 결과를 덮어쓰지 않고 UTC timestamp 디렉터리를 만듭니다. 사용 가능한 인자는 다음으로 확인하세요.

```bash
python scripts/evaluate_vost_base_roundtrip.py --help
python scripts/vost_subset.py --help
```

## 업로드 범위와 데이터

VOST RGB/annotation, SAM 2 checkpoint, 예측 mask, 실행별 raw output은 Git에 올리지 않습니다. `.gitignore`가 `data/`, `outputs/`, `.external/`을 제외합니다. GitHub에서 재현할 때는 각 사용자가 [VOST 공식 데이터 페이지](https://www.vostdataset.org/data.html)의 조건에 따라 데이터를 내려받아야 합니다. VOST 데이터는 CC BY-NC-SA 4.0 조건이므로 데이터를 이 저장소에 재배포하지 마세요.

`scripts/setup_vost_evaluator.sh`는 공식 [TRI-ML/VOST](https://github.com/TRI-ML/VOST) evaluator를 고정 commit `fe274574cb03c8a3ea83e121dd76e20b703781fd`으로 `.external/` 아래에 준비합니다. `reports/`에는 결과 해석에 필요한 작은 요약만 보관하고, 상세 prediction/output은 로컬 `outputs/`에 둡니다.

이 저장소에는 handoff 공통 모듈과 기존 연구용 코드도 포함되어 있습니다. 이번 VOST 실행의 진입점은 `scripts/evaluate_vost_base_roundtrip.py`, dataset loader/evaluator는 `src/vos_memory_inspector/vost_roundtrip.py`입니다.

공개 전 확인: `src/`, `scripts/`, `tests/`, `docs/`, `manifests/`에는 기존 DAVIS/MOSE/LVOS 연구 코드·자료도 남아 있으며 VOST 실행에는 모두 필요하지 않습니다. 코드 license 파일은 아직 없으므로 공개 범위와 license를 정한 뒤 업로드하세요. 위 `.gitignore`는 데이터·checkpoint·외부 evaluator·실행별 output을 제외하지만 기존 소스와 문서는 기본적으로 포함됩니다.
