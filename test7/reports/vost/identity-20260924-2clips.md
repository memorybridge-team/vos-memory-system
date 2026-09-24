# VOST Base+ identity handoff — 2026-09-24

## Scope

Two complete VOST validation sequences, first-frame mask prompting, midpoint state handoff, and continuation on a fresh Target predictor. Source and Target use the same SAM 2.1 Base+ checkpoint with `DirectCopyTranslator`; no learned cross-model translation is tested.

| Sequence | Frames | Switch (internal index / original ID) | Post-switch J&F rows | Mean post-switch delta | Result |
|---|---:|---:|---:|---:|---|
| `555_tear_aluminium_foil` | 51 | 25 / 300 | 25 | 0.000000 | passed |
| `6922_split_paper` | 78 | 39 / 234 | 38 | 0.000000 | passed |

## Recorded metrics

- Official VOST aggregate, native and transferred: `J = 0.5425676`, `J_last = 0.6239493`.
- Official aggregate delta: `ΔJ = 0`, `ΔJ_last = 0`.
- Supplemental post-switch J&F: 63 rows, mean `0.6273545`, mean delta `0`, max absolute delta `0`.
- All post-switch binary masks were equal; max logit error was `0`; target injection made `0` backbone calls.
- Device `cpu`, seed `7`; dataset payload 35,989,815 bytes.

The official aggregate and supplemental per-frame curves use different frame inclusion rules; do not treat their means as interchangeable. The official evaluator excludes the first and last sequence frames from its aggregate. The two-sequence result is a code-path identity check, not a dataset-level accuracy claim.

## Reproduction pins

- SAM 2 commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Checkpoint SHA-256: `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`
- Official VOST evaluator commit: `fe274574cb03c8a3ea83e121dd76e20b703781fd`
- Dataset manifest SHA-256: `2c783b3efca83d8d78e06a5dd4d2f61e0a696b6db1c3d7b4da4d7d3bc8cda8e2`

The full local run report, per-frame CSV/SVG curves, and prediction masks remain under `outputs/vost_base_roundtrip/identity-20260924-2clips/` and are ignored by Git. The dataset itself is not included; see [`plan.md`](../../plan.md) for the download and evaluation commands.
