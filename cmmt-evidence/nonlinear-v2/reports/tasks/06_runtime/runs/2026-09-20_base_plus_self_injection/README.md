# Base+ self-injection runtime validation

- Date: 2026-09-20 (Asia/Seoul)
- Project commit: `32dc1400b5b3b2cbbbbac11f4780010358a5f83b`
- SAM 2 commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Device: NVIDIA RTX 2000 Ada Generation, 16,380 MiB
- PyTorch/CUDA: 2.8.0+cu128 / CUDA 12.8
- Dataset: DAVIS 2017 `walking`
- Object ID: `1`
- Switch frame: `10`
- Seed: `7`

## Command

```bash
bash scripts/runpod_base_plus_roundtrip.sh \
  /workspace/vos-memory-translator-nonlinear \
  /workspace/vos-memory-translator-nonlinear/data/DAVIS \
  walking 1 10 base_plus_roundtrip_walking_f10_20260920
```

## Result

- Future continuation: frames 11-71 (61 frames)
- Mean binary IoU between native and injected continuation: `1.0`
- Mean MSE: `0.0`
- Maximum absolute error: `0.0`
- Backbone calls during injection: `0`
- Wall time: `132.53 s`
- Peak CUDA memory: `974,874,624 bytes` (about 0.91 GiB)

The acceptance gate passed. This confirms that, for one object with a first-frame
mask prompt on this sequence, the exported Base+ state can be assembled into a
fresh Base+ predictor without replaying past RGB frames and produces identical
future predictions.

## Scope of the evidence

This is runtime evidence for the state export/injection boundary, not evidence
that Small-to-Base+ translation succeeds. The Small/Base+ inventory, paired
dump, and final I/O policy freeze were completed separately, so task 02 is now
complete. Multi-object, late-prompt, absence/reappearance, prompt-correction,
and cross-model target-injection behavior belong to task 06.

Raw `report.json`, `status.txt`, and `stdout.log` are copied from the RunPod run
into this directory.
