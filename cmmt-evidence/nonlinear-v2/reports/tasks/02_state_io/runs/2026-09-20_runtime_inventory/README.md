# Small/Base+ runtime state inventory

- Date: 2026-09-20 (Asia/Seoul)
- Dataset: DAVIS 2017 `walking`
- Object ID: `1`
- Switch frame: `10`
- Seed: `7`
- Source: SAM 2.1 Small
- Target: SAM 2.1 Base+

## Confirmed runtime boundary

The two checkpoints produced the same canonical boundary for this paired case.

| Field | Small | Base+ | Policy |
|---|---|---|---|
| `spatial_memory` | `[1,1,11,64,64,64]`, bfloat16 | `[1,1,11,64,64,64]`, bfloat16 | translate |
| `object_pointer` | `[1,1,11,256]`, float32 | `[1,1,11,256]`, float32 | translate |
| `presence_logits` | `[1,1,11,1]`, float32 | `[1,1,11,1]`, float32 | calibrate/ablate |
| `frame_indices` | `[1,1,11]`, int64 | `[1,1,11]`, int64 | preserve |
| `slot_order` | `[1,1,11]`, int64 | `[1,1,11]`, int64 | preserve |
| `is_conditioning` | `[1,1,11]`, bool | `[1,1,11]`, bool | preserve |
| `validity` | `[1,1,11]`, bool | `[1,1,11]`, bool | preserve |

Both sides contained 11 valid records and 5,778,476 bytes of continuous
translator-boundary data. The prepared cache contains 11 source prefix frames
and 61 target-oracle future frames.

## Paired artifact

- Remote cache SHA-256: `5e9bca17217d522335acf80a454834cf2beab22d4dd8f6204fcd221d2bf1a5f0`
- Remote cache size: `166,882,485 bytes`
- Preparation wall time: `72.23 s`
- Peak CUDA memory: `982,584,320 bytes`

The 167 MB `.pt` cache remains on the RunPod network volume and is not committed
to Git. The repository stores its checksum, contract report, and tensor
inventories so the artifact can be verified without bloating Git history.

## Interpretation

This confirms the runtime I/O shape and dtype contract for one real paired
Small/Base+ example. It does not prove cross-model semantic equivalence or
translator quality; those require training and downstream continuation metrics.
