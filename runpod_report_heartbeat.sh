#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_FILE="${REPORT_FILE:-$ROOT_DIR/RUNPOD_PROGRESS.md}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"

append_status() {
  local now local_size
  now="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  local_size="$(du -sh "$ROOT_DIR/local_data" 2>/dev/null | awk '{print $1}' || true)"
  {
    printf '\n- %s — heartbeat: local_data=%s; RunPod job status is recorded from the active SSH terminal.\n' "$now" "${local_size:-unknown}"
  } >> "$REPORT_FILE"
}

while :; do
  append_status
  sleep "$INTERVAL_SECONDS"
done
