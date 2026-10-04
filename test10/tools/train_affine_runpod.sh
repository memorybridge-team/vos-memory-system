#!/usr/bin/env bash
# Train the comparison Affine on the RunPod volume that holds the reference Transformer run.
# Paths default to the 2a57d85-corrected run; override with environment variables.
set -euo pipefail

TEST10_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REFERENCE_RUN="${REFERENCE_RUN:-/workspace/CMMT-lvos-isolated/2a57d85-corrected-20261004T022603Z}"
OUTPUT_DIR="${OUTPUT_DIR:-$TEST10_ROOT/training/affine_comparable_v1}"
PYTHON="${PYTHON:-python}"
: "${TEST10_TRANSLATOR_REPO:?set to the translator checkout used by test10 evaluation (not 2a57d85; its linear preset cannot be built)}"
export TEST10_TRANSLATOR_REPO

if [ -e "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR")" ]; then
  echo "output directory is not empty: $OUTPUT_DIR" >&2
  exit 1
fi

# raw_index.json stores the original /workspace cache paths, so no --pair-root is needed here.
"$PYTHON" "$TEST10_ROOT/comparable_training.py" train \
  --index "$REFERENCE_RUN/index/raw_index.json" \
  --transformer-run "$REFERENCE_RUN/training" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda "$@"
