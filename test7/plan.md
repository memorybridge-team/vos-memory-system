# VOST 소규모 Base+ → Base+ handoff 재현 계획

## 목적과 주장 범위

이 실험은 SAM 2.1 Base+가 VOST 영상에서 같은 체크포인트의 새 Target predictor에 memory를 넘긴 뒤, 끊김 없는 native 추론과 같은 결과를 내는지 확인한다. 이는 state assembly/injection 경로의 **same-model identity diagnostic**이다. translator 학습, Small→Base+ cross-model translation, 데이터셋 전체 성능은 검증하지 않는다.

## 고정 프로토콜

- Dataset/split: VOST `val`; 완전한 validation sequence 2개만 사용한다.
- Sequence: `555_tear_aluminium_foil` (51 sampled frames), `6922_split_paper` (78 sampled frames). 선정된 RGB+annotation은 약 36 MB다.
- Prompt: 각 sequence 첫 annotated frame의 모든 객체 ID에 대해 GT mask를 한 번 제공한다. 이후 correction/prompt는 없다.
- Switch: sequence 순서상 중간 index (`floor(N/2)`). Source는 switch까지 처리하고, Target은 switch + 1부터 continuation한다. sequence 내 `frame_idx`와 원래 VOST frame ID를 별도 기록한다.
- Transfer: Source와 Target 모두 SAM 2.1 Base+ 및 동일 checkpoint를 사용한다. `DirectCopyTranslator`가 canonical memory를 identity 방식으로 전달하며 학습은 없다.
- Target 초기화: SAM 2 API상 raw RGB frame 전체가 `inference_state["images"]`에 로드될 수 있다. 이는 전체 frame을 미리 추론해 memory/output을 계산했다는 뜻은 아니다. handoff 주입 중 과거 frame backbone 호출이 0회인지 별도로 계측한다.
- Metrics: per-frame J/F는 공식 VOST metric 함수로 계산한다. 공식 evaluator의 aggregate `J`와 `J_last`를 native/transferred prediction 각각에 적용한다. 보조 J&F 곡선은 모든 frame을 포함하며 공식 aggregate와 집계 범위가 다르다.
- Pass gate: injected state metadata/IDs 유효, injection 중 backbone 호출 0회, 이후 예상 frame 수만큼 추론, post-switch binary masks 동일, logits와 J/F 차이가 각각 `1e-6` 이하.

## 재현 조건

- SAM 2 checkout: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Config: `configs/sam2.1/sam2.1_hiera_b+.yaml`
- Checkpoint: `sam2.1_hiera_base_plus.pt`, SHA-256 `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`
- Official VOST evaluator: commit `fe274574cb03c8a3ea83e121dd76e20b703781fd`
- Device/seed in recorded run: CPU / 7
- Dataset entry별 byte count와 SHA-256은 내려받을 때 `data/vost_subset/val/download_manifest.json`에 남고, 실행 전 전수 검증한다.

## 실행

저장소 루트에서 SAM 2 의존성이 설치된 Python 환경을 사용한다.

```bash
python -m pip install -e ".[dev,vost-eval]"
python -m pip install -e ../sam2
bash scripts/setup_vost_evaluator.sh
python scripts/vost_subset.py --videos 555_tear_aluminium_foil 6922_split_paper
python scripts/evaluate_vost_base_roundtrip.py --check-data-only
python scripts/evaluate_vost_base_roundtrip.py \
  --sam2-repo ../sam2 \
  --checkpoint ../sam2/checkpoints/sam2.1_hiera_base_plus.pt \
  --device auto
```

실행은 `outputs/vost_base_roundtrip/<UTC timestamp>/`에 report, 영상별 CSV/SVG, native/transferred prediction을 쓴다. Raw dataset, checkpoint, predictions는 Git에 추가하지 않는다.

## 기록된 실행 결과 (2026-09-24)

- Status: `passed`; videos 2, frames 129, downloaded RGB+annotation 35,989,815 bytes.
- Official VOST metrics: native/transferred 모두 `J=0.5425676`, `J_last=0.6239493`; aggregate delta는 각각 `0`.
- Post-switch supplemental J&F: 63 object/frame rows, mean `0.6273545`; native/transferred mean delta `0`, maximum absolute delta `0`.
- 두 영상 모두 post-switch mask가 모두 동일했고 max logit error `0`; injection 중 backbone 호출 `0`회.
- 영상별 switch: foil 내부 index 25 (원래 frame ID 300), paper 내부 index 39 (원래 frame ID 234).

이 수치는 두 validation sequence에 대한 same-checkpoint 재현 결과다. 일반화, 실제 모델 전환, 학습 translator 품질의 근거로 해석하지 않는다. 상세값은 [`reports/vost/identity-20260924-2clips.md`](reports/vost/identity-20260924-2clips.md)와 실행 report JSON을 참조한다.

## 주요 코드

- `scripts/vost_subset.py`: 공식 VOST ZIP에서 선택한 entry만 HTTP Range로 다운로드하고 manifest/hash 생성.
- `scripts/setup_vost_evaluator.sh`: 공식 evaluator의 고정 commit checkout.
- `scripts/evaluate_vost_base_roundtrip.py`: 데이터 검사와 실험 실행 CLI.
- `src/vos_memory_inspector/vost_roundtrip.py`: VOST split/annotation loader, 공식 평가기, 실행 report.
- `src/vos_memory_inspector/base_roundtrip.py`: dataset과 무관한 predictor handoff stream/scoring/curve helper 및 LVOS 실험 helper.
