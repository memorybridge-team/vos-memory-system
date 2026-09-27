# SAM 2.1 Small → Base+ Memory Preconditioning Report

## A. Environment

- SAM2 upstream commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- PyTorch/CUDA: `2.8.0+cu128` / `12.8`
- GPU: `NVIDIA GeForce RTX 4080 SUPER`
- Inference/translator dtype: `bfloat16` / `float32`
- Small checkpoint SHA-256: `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`
- Base+ checkpoint SHA-256: `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`

## B. Verified SAM2.1 memory path

See `code_audit.md`. Target positional encodings and temporal assembly remain target-native.

## C. Checkpoint audit

See `checkpoint_audit.md`, `checkpoint_audit.json`, and `checkpoint_matrices.pt`.

## D. Candidate transforms

Spatial: M0 raw, M1 standardization, M4 consumer-aware, M2 ZCA, M3 output-projection SVD. Pointer: P0 raw, P1 standardization, P2 anchor, P2+P4 anchor/consumer, P3 final-linear.

## E. Round-trip sanity

- spatial/M0_raw: relative L2=0.000e+00, finite=True, deployment finite=True.
- spatial/M1_standardized: relative L2=3.793e-08, finite=True, deployment finite=True.
- spatial/M4_consumer: relative L2=1.540e-07, finite=True, deployment finite=True.
- spatial/M2_zca: relative L2=1.249e-06, finite=True, deployment finite=True.
- spatial/M3_out_proj_svd: relative L2=8.032e-07, finite=True, deployment finite=True.
- pointer/P0_raw: relative L2=0.000e+00, finite=True, deployment finite=True.
- pointer/P1_standardized: relative L2=4.045e-09, finite=True, deployment finite=True.
- pointer/P2_anchor: relative L2=2.395e-08, finite=True, deployment finite=True.
- pointer/P2_P4_anchor_consumer: relative L2=1.506e-07, finite=True, deployment finite=True.
- pointer/P3_final_linear: relative L2=8.226e-07, finite=True, deployment finite=True.

## F. Aligned-pair representation results

- `tiny_lvos_M0_P0_mlp_aligned_fit.json`: records=16, spatial MSE=0.07191486284136772, pointer MSE=0.008446514738352562.

## G. Native-pair representation results

- `tiny_lvos_M0_P0_mlp_native_fit.json`: records=16, spatial MSE=0.06980911828577518, pointer MSE=0.0011050928421241224.

## H. Consumer/attention results

The same-query K/V and attention diagnostic is implemented; held-out results are pending.

## I. Actual no-replay continuation

Held-out no-replay continuation has not been run; no method conclusion is made.

## J. Runtime and transfer cost

Pending held-out no-replay runs. Reference generation and target initialization are reported separately by the evaluator.

## K. Interpretation

The completed numbers are one-video fit sanity checks. They show that the pipeline trains, but they do not compare canonicalization candidates or establish VOS quality. The final decision remains held-out native Small-prefix → Base+ no-replay suffix performance; lower state error alone is not treated as success.
