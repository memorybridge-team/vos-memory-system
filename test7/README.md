# VOST Base+ handoff diagnostic

이 저장소의 현재 재현 실험은 **VOST validation 영상에서 SAM 2.1 Base+ → Base+ 상태 이전이 같은 추론을 재현하는지** 확인합니다. 학습된 Small→Base+ translator의 성능을 검증하는 실험은 아닙니다.

The current reproducible experiment checks whether a SAM 2.1 Base+ → Base+ state handoff reproduces uninterrupted inference on a small VOST validation subset. It does **not** evaluate a trained Small→Base+ translator.

## 실험 요약

- VOST val의 전체 영상 2개: `555_tear_aluminium_foil` (51 frames), `6922_split_paper` (78 frames); 이미지·마스크 약 36 MB.
- 각 영상 첫 annotated frame의 모든 객체 mask만 prompt로 사용하고, 영상 중간에서 handoff합니다. Target은 switch 다음 frame부터 추론합니다.
- 같은 Base+ checkpoint를 Source와 Target에 사용하고, `DirectCopyTranslator`로 memory state를 그대로 전달합니다. 학습은 하지 않습니다.
- 성공 조건: 주입 도중 과거 영상 backbone 재계산 없음, 이후 logits/masks/J&F가 native 추론과 허용오차 내에서 일치.
- SAM 2 Target state에는 raw RGB frame들이 로드될 수 있지만, 이는 모든 frame을 미리 추론해 memory를 만든다는 뜻은 아닙니다. 이 실험은 handoff 주입 시 과거 frame backbone 호출이 0회인지 계측합니다.
- 현재 기록 결과: 공식 VOST aggregate `J=0.54257`, `J_last=0.62395`; native와 transferred 간 차이는 둘 다 `0`. 두 영상의 사후 handoff 예측 mask와 logits도 일치했습니다.

영상 두 개만 쓴 제한된 same-checkpoint 진단이므로, 데이터셋 전체 성능이나 cross-model/nonlinear translation의 효용으로 일반화하면 안 됩니다. 프로토콜과 결과 해석은 [`plan.md`](plan.md), 실행 기록은 [`reports/vost/identity-20260924-2clips.md`](reports/vost/identity-20260924-2clips.md)에 있습니다.

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
