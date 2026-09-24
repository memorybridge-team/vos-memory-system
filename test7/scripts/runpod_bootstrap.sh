#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/runpod_bootstrap.sh /workspace/vos-memory-translator-nonlinear"
  exit 2
fi

project_dir="$(cd "$1" && pwd)"
sam2_dir="${project_dir}/.external/sam2"
checkpoint_dir="${project_dir}/checkpoints"
venv_dir="${project_dir}/.venv"
if [[ -f "${project_dir}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${project_dir}/.env"
  set +a
fi
sam2_commit="${SAM2_UPSTREAM_COMMIT:-2b90b9f5ceec907a1c18123530e92e794ad901a4}"

# RunPod's base Python can be externally managed (PEP 668).  Keep every
# project package in a venv while reusing the image's CUDA-enabled PyTorch.
if [[ ! -x "${venv_dir}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${venv_dir}"
fi
python_bin="${venv_dir}/bin/python"

"${python_bin}" -m pip install --upgrade pip
"${python_bin}" -m pip install -e "${project_dir}[dev]"

if [[ ! -d "${sam2_dir}/.git" ]]; then
  git clone https://github.com/facebookresearch/sam2.git "${sam2_dir}"
fi
git -C "${sam2_dir}" fetch origin "${sam2_commit}"
git -C "${sam2_dir}" checkout --detach "${sam2_commit}"
# Do not let pip build isolation fetch a different CUDA build of PyTorch than
# the CUDA-enabled one bundled by the RunPod image.
SAM2_BUILD_ALLOW_ERRORS=0 "${python_bin}" -m pip install --no-build-isolation -v -e "${sam2_dir}"

mkdir -p "${checkpoint_dir}"
for checkpoint in sam2.1_hiera_small.pt sam2.1_hiera_base_plus.pt; do
  if [[ ! -f "${checkpoint_dir}/${checkpoint}" ]]; then
    curl -fL \
      "https://dl.fbaipublicfiles.com/segment_anything_2/092824/${checkpoint}" \
      -o "${checkpoint_dir}/${checkpoint}"
  fi
done

"${python_bin}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

inventory_dir="${project_dir}/outputs/inventory"
mkdir -p "${inventory_dir}"
inventory_path="${inventory_dir}/bootstrap_$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  echo "timestamp_utc=$(date -u --iso-8601=seconds)"
  echo "project_dir=${project_dir}"
  echo "sam2_commit=$(git -C "${sam2_dir}" rev-parse HEAD)"
  echo "python=$(${python_bin} --version)"
  "${python_bin}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY
  df -h "${project_dir}"
  sha256sum "${checkpoint_dir}/sam2.1_hiera_small.pt"
  sha256sum "${checkpoint_dir}/sam2.1_hiera_base_plus.pt"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} | tee "${inventory_path}"

echo "Bootstrap inventory: ${inventory_path}"
