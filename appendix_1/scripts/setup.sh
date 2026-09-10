#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPENDIX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"
PROFILE="${1:-cu124}"
PYTHON_BIN="${PYTHON:-python3}"

case "${PROFILE}" in
  cu124|cu121|cpu) TORCH_INDEX="https://download.pytorch.org/whl/${PROFILE}" ;;
  *) echo "Usage: ./scripts/setup.sh [cu124|cu121|cpu]" >&2; exit 2 ;;
esac

"${PYTHON_BIN}" -m venv "${APPENDIX_DIR}/.venv"
"${APPENDIX_DIR}/.venv/bin/python" -m pip install --upgrade pip
"${APPENDIX_DIR}/.venv/bin/python" -m pip install \
  torch==2.5.1 torchvision==0.20.1 --index-url "${TORCH_INDEX}"
"${APPENDIX_DIR}/.venv/bin/python" -m pip install \
  -r "${APPENDIX_DIR}/requirements.txt"

if [[ ! -d "${APPENDIX_DIR}/vendor/sam2/.git" ]]; then
  mkdir -p "${APPENDIX_DIR}/vendor"
  git clone https://github.com/facebookresearch/sam2.git "${APPENDIX_DIR}/vendor/sam2"
fi
git -C "${APPENDIX_DIR}/vendor/sam2" fetch origin "${SAM2_COMMIT}"
git -C "${APPENDIX_DIR}/vendor/sam2" checkout --detach "${SAM2_COMMIT}"
SAM2_BUILD_CUDA=0 "${APPENDIX_DIR}/.venv/bin/python" -m pip install \
  --no-build-isolation --no-deps -e "${APPENDIX_DIR}/vendor/sam2"

echo "Environment ready for ${PROFILE}. Check CUDA visibility with the README command."
