#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/runpod_preflight.sh /workspace/vos-memory-translator-nonlinear"
  exit 2
fi

project_dir="$(cd "$1" && pwd)"
workspace_dir="/workspace"
required_free_gb=40

if [[ ! -d "${workspace_dir}" ]]; then
  echo "Expected RunPod Network Volume mount is missing: ${workspace_dir}" >&2
  exit 1
fi
if [[ "${project_dir}" != "${workspace_dir}"/* ]]; then
  echo "Project must be located under ${workspace_dir}: ${project_dir}" >&2
  exit 1
fi

available_kb="$(df -Pk "${workspace_dir}" | awk 'NR == 2 {print $4}')"
required_kb=$((required_free_gb * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "Need at least ${required_free_gb} GiB free on ${workspace_dir}; found $((available_kb / 1024 / 1024)) GiB." >&2
  exit 1
fi

for path in \
  "${project_dir}/.external/sam2" \
  "${project_dir}/checkpoints/sam2.1_hiera_small.pt" \
  "${project_dir}/checkpoints/sam2.1_hiera_base_plus.pt" \
  "${project_dir}/.venv/bin/python"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing prerequisite: ${path}" >&2
    exit 1
  fi
done

report_dir="${project_dir}/outputs/inventory"
mkdir -p "${report_dir}"
report_path="${report_dir}/preflight_$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  echo "timestamp_utc=$(date -u --iso-8601=seconds)"
  echo "project_dir=${project_dir}"
  echo "workspace_free_gib=$((available_kb / 1024 / 1024))"
  echo "required_free_gib=${required_free_gb}"
  git -C "${project_dir}" rev-parse HEAD
  git -C "${project_dir}/.external/sam2" rev-parse HEAD
  sha256sum "${project_dir}/checkpoints/sam2.1_hiera_small.pt"
  sha256sum "${project_dir}/checkpoints/sam2.1_hiera_base_plus.pt"
  nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
  df -h "${workspace_dir}"
} | tee "${report_path}"

echo "Preflight passed. Inventory: ${report_path}"
