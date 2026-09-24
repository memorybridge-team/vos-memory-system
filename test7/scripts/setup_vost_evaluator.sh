#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ROOT}/.external/vost-evaluation"
URL="https://github.com/TRI-ML/VOST.git"
EXPECTED_COMMIT="fe274574cb03c8a3ea83e121dd76e20b703781fd"

if [[ -e "${TARGET}" ]]; then
  if [[ -d "${TARGET}/.git" ]]; then
    ACTUAL_COMMIT="$(git -C "${TARGET}" rev-parse HEAD 2>/dev/null || true)"
    if [[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]]; then
      printf 'VOST evaluator already at pinned commit %s\n' "${ACTUAL_COMMIT}"
      exit 0
    fi
  fi
  printf 'Refusing to overwrite existing evaluator path: %s\n' "${TARGET}" >&2
  exit 1
fi

mkdir -p "$(dirname "${TARGET}")"
git clone --no-checkout "${URL}" "${TARGET}"
git -C "${TARGET}" checkout --detach "${EXPECTED_COMMIT}"
ACTUAL_COMMIT="$(git -C "${TARGET}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  printf 'Expected evaluator commit %s, got %s\n' "${EXPECTED_COMMIT}" "${ACTUAL_COMMIT}" >&2
  exit 1
fi
printf 'Prepared official VOST evaluator at %s\n' "${ACTUAL_COMMIT}"
