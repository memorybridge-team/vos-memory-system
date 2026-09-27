# SAM 2.1 Small → Base+ memory preconditioning

SAM 2.1 Small이 switch 시점까지 만든 객체별 temporal memory(`maskmem_features`, `obj_ptr`)를 고정 canonical transform과 작은 translator head로 변환해 Base+에 주입하고, Base+가 과거 프레임을 다시 처리하지 않고 `switch+1`부터 추적을 이어갈 수 있는지 검증하는 연구 코드입니다.

- 변환 후보: spatial `M0` raw, `M1` standardization, `M2` ZCA, `M3` output-projection SVD, `M4` consumer-aware / pointer `P0` raw, `P1` standardization, `P2` no-object anchor, `P2+P4` anchor+consumer, `P3` final-linear
- 학습: aligned pair 학습 → 같은 고정 transform을 유지한 native pair 미세조정. Transform 통계는 aligned fit split에서만 계산
- 최종 판단 기준: held-out native Small prefix → Base+ no-replay suffix J&F. State MSE와 K/V·attention 유사도는 진단 지표

자세한 프로토콜은 [docs/design/memory_preconditioning_protocol.md](docs/design/memory_preconditioning_protocol.md)에 있습니다.

## 현재 상태

- 완료: SAM 2.1 source/checkpoint audit, LVOS v2 train 80/10/10 source-group split, 전체 transform suite round-trip, LVOS 한 영상(`0ClBYzYm`)에 대한 `M0_raw`/`P0_raw` MLP fit sanity check
- 미완료: 후보 간 비교, VOST 학습, held-out no-replay continuation 평가. 현재 수치로는 방법에 대한 결론을 내리지 않습니다 ([report.md](experiments/sam21_small_to_baseplus_precondition/report.md))

## 구성

| 경로 | 내용 |
|---|---|
| `src/vos_memory_inspector/` | 상태 추출·주입, transform, translator, 평가 코드 (`precondition_*.py`, `preconditioning.py`가 이 실험의 핵심) |
| `tests/` | synthetic 회귀 테스트 (GPU·데이터·SAM 2 불필요) |
| `configs/sam21_small_base_precondition.json` | 실험 설정 기록 (참고용, 명령이 읽지 않음) |
| `docs/design/` | 실험 프로토콜 |
| `experiments/sam21_small_to_baseplus_precondition/` | audit, split manifest, transform suite, tiny translator, 평가·보고서 |

`pyproject.toml`의 다른 `sam2-*`/`cmmt-*` 명령은 상위 연구 프로젝트에서 가져온 DAVIS/MOSE/LVOS manifest, round-trip, baseline 도구입니다.

## 설치

Python 3.10 이상이 필요합니다.

```bash
pip install -e ".[dev]"      # gdown이 필요하면 ".[dev,data]"
pytest
ruff check src tests
```

실제 실험에는 다음 외부 구성요소가 필요하며 이 저장소에는 포함되지 않습니다.

- [facebookresearch/sam2](https://github.com/facebookresearch/sam2) checkout, commit `2b90b9f5ceec907a1c18123530e92e794ad901a4` 고정 (다른 commit이면 실행 전에 중단됩니다). SAM 2 의존성은 해당 checkout에서 설치합니다.
- SAM 2.1 `sam2.1_hiera_small.pt`, `sam2.1_hiera_base_plus.pt` checkpoint. 기대 SHA-256은 [checkpoint_audit.md](experiments/sam21_small_to_baseplus_precondition/checkpoint_audit.md)에 있습니다.
- J&F 평가용 [davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation) checkout
- LVOS v2, VOST 데이터. `download` 명령은 기록된 SHA-256·크기와 다른 archive를 압축 해제 전에 거부합니다.

## 실행 순서

모든 단계는 `cmmt-sam21-precondition <단계>`로 실행하며, 경로는 모두 명령 인자로 전달합니다. 아래 `SAM2`, `CKPT`, `DATA`, `EXP`는 예시 변수입니다.

```bash
EXP=experiments/sam21_small_to_baseplus_precondition
MODELS="--sam2-repo $SAM2 \
  --source-config configs/sam2.1/sam2.1_hiera_s.yaml --source-checkpoint $CKPT/sam2.1_hiera_small.pt \
  --target-config configs/sam2.1/sam2.1_hiera_b+.yaml --target-checkpoint $CKPT/sam2.1_hiera_base_plus.pt"

cmmt-sam21-precondition download lvos_v2_train --download-dir $DATA/downloads --extract-dir $DATA/LVOS_V2
cmmt-sam21-precondition audit $MODELS --output-dir $EXP --device cuda
cmmt-sam21-precondition split --dataset-root $DATA/LVOS_V2 --dataset LVOS_V2 --release v2 \
  --output $EXP/manifests/lvos_v2_train_seed7.json
cmmt-sam21-precondition extract $MODELS --manifest $EXP/manifests/lvos_v2_train_seed7.json \
  --dataset-root $DATA/LVOS_V2 --output-root $EXP/state_pairs --tiny-overfit
cmmt-sam21-precondition fit-transforms --aligned-fit-bank $EXP/state_pairs/LVOS_V2/aligned/fit/index.json \
  --checkpoint-matrices $EXP/checkpoint_matrices.pt \
  --fit-manifest $EXP/manifests/lvos_v2_train_seed7.json --output $EXP/roundtrip/transform_suite.pt
cmmt-sam21-precondition train --transform-suite $EXP/roundtrip/transform_suite.pt \
  --aligned-bank $EXP/state_pairs/LVOS_V2/aligned/fit/index.json \
  --native-bank $EXP/state_pairs/LVOS_V2/native/fit/index.json \
  --spatial-method M0_raw --pointer-method P0_raw \
  --output-dir $EXP/checkpoints/translators/tiny_lvos_M0_P0_mlp_seed7
cmmt-sam21-precondition evaluate \
  --artifact $EXP/checkpoints/translators/tiny_lvos_M0_P0_mlp_seed7/translator_native_finetuned.pt \
  --pair-bank $EXP/state_pairs/LVOS_V2/native/fit/index.json --pair-type native --split fit \
  --output $EXP/evaluations/tiny_lvos_M0_P0_mlp_native_fit.json
cmmt-sam21-precondition report --experiment-root $EXP
```

- `fit-transforms`와 `train`은 fit split 이외의 record나 잘못된 pair type이 섞이면 중단합니다. `evaluate`도 `--split`/`--pair-type`과 bank가 다르면 중단합니다.
- 실제 no-replay 평가는 `evaluate --case-cache ...`로 실행하고, prefix backbone 호출이 0회가 아니면 실패합니다. `--help`에서 필요한 인자를 확인할 수 있습니다.
- 산출물 JSON에는 로컬 절대경로 대신 저장소 기준 상대경로나 파일 이름과 SHA-256을 기록합니다.

## 데이터와 라이선스

- 영상·mask 원본, 추출된 `state_pairs/` bank(약 1.2 GB), SAM 2 checkpoint는 포함하지 않습니다. Split manifest에는 video/frame ID만 있습니다.
- LVOS v2: 비상업 연구용, annotation CC BY 4.0. VOST: CC BY-NC-SA 4.0. 포함된 tiny translator checkpoint는 LVOS v2 train 영상 하나로 학습했습니다.
- `checkpoint_matrices.pt`와 `roundtrip/transform_suite.pt`에는 SAM 2.1 checkpoint(Apache-2.0)에서 추출한 가중치 일부가 들어 있습니다. [NOTICE](NOTICE)를 참고하세요.
- 이 저장소 코드의 라이선스는 아직 지정하지 않았습니다.
