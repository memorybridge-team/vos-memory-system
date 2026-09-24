#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: bash scripts/runpod_base_plus_roundtrip.sh PROJECT_DIR DAVIS_ROOT SEQUENCE OBJECT_ID SWITCH_FRAME [RUN_ID]"
  exit 2
fi

project_dir="$(cd "$1" && pwd)"
davis_root="$(cd "$2" && pwd)"
sequence="$3"
object_id="$4"
switch_frame="$5"
run_id="${6:-base_plus_roundtrip_$(date -u +%Y%m%dT%H%M%SZ)}"

video_dir="${davis_root}/JPEGImages/480p/${sequence}"
prompt_mask="${davis_root}/Annotations/480p/${sequence}/00000.png"
python_bin="${project_dir}/.venv/bin/python"
runner="${project_dir}/.venv/bin/sam2-roundtrip-smoke"
output_dir="${project_dir}/outputs/roundtrip/${run_id}"
status_path="${output_dir}/status.txt"
report_path="${output_dir}/report.json"

if [[ ! -x "${runner}" ]]; then
  echo "Missing runner: ${runner}. Run runpod_bootstrap.sh first." >&2
  exit 1
fi
if [[ ! -d "${video_dir}" || ! -f "${prompt_mask}" ]]; then
  echo "DAVIS sequence or first prompt mask is missing for ${sequence}." >&2
  exit 1
fi

mkdir -p "${output_dir}"
printf 'state=running\nstarted_at=%s\n' "$(date -u --iso-8601=seconds)" > "${status_path}"
trap 'code=$?; printf "state=failed\\nexit_code=%s\\nfinished_at=%s\\n" "${code}" "$(date -u --iso-8601=seconds)" > "${status_path}"; exit "${code}"' ERR

"${python_bin}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; do not run a CPU round-trip by accident.")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"cuda={torch.version.cuda}")
PY

"${runner}" \
  --sam2-repo "${project_dir}/.external/sam2" \
  --config configs/sam2.1/sam2.1_hiera_b+.yaml \
  --checkpoint "${project_dir}/checkpoints/sam2.1_hiera_base_plus.pt" \
  --model-id sam2.1-base+ \
  --video-dir "${video_dir}" \
  --prompt-mask "${prompt_mask}" \
  --object-id "${object_id}" \
  --switch-frame "${switch_frame}" \
  --device cuda \
  --json "${report_path}" | tee "${output_dir}/stdout.log"

"${python_bin}" - "${report_path}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
comparison = report["comparison"]
if comparison["mean_binary_iou"] < 0.999:
    raise SystemExit(f"round-trip failed: mean_binary_iou={comparison['mean_binary_iou']}")
if comparison["max_abs_error"] > 1e-4:
    raise SystemExit(f"round-trip failed: max_abs_error={comparison['max_abs_error']}")
if report["backbone_calls_during_injection"] != 0:
    raise SystemExit("round-trip failed: injection replayed a past frame")
print("round-trip acceptance gate passed")
PY

printf 'state=completed\nfinished_at=%s\nreport=%s\n' \
  "$(date -u --iso-8601=seconds)" "${report_path}" > "${status_path}"
echo "Completed: ${output_dir}"
