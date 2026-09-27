# SAM 2.1 Small → Base+ memory preconditioning

이 디렉터리는 `cmmt-sam21-precondition`의 산출물 루트다. Dataset, SAM 2 checkpoint,
추출된 state-pair bank(`state_pairs/`)는 Git에 넣지 않는다.

```text
sam21_small_to_baseplus_precondition/
├── code_audit.md                 # 검증한 SAM 2.1 memory 경로 (SAM 2 checkout 기준 상대경로)
├── checkpoint_audit.md
├── checkpoint_audit.json         # spectra, 환경, checkpoint 파일 이름·SHA-256
├── checkpoint_matrices.pt        # transform fitting에 쓰는 checkpoint 행렬
├── manifests/                    # LVOS v2 train 80/10/10 source-group split
├── roundtrip/                    # transform suite + SHA-256 sidecar
├── checkpoints/translators/      # tiny translator artifact와 training_report.json
├── evaluations/                  # pair-bank state 평가 JSON
└── report.md
```

명령 순서는 `download → audit → split → extract → fit-transforms → train →
evaluate → report`다. `evaluate`는 pair-bank state 평가와 prepared case를 이용한 실제
no-replay continuation을 모두 지원한다. 최종 판단에는 후자만 사용한다.
