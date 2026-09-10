#!/usr/bin/env bash
set -Eeuo pipefail

RUNPOD_HOST="${RUNPOD_HOST:-}"
RUNPOD_PORT="${RUNPOD_PORT:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/runpod_ed25519}"

if [[ -z "$RUNPOD_HOST" || -z "$RUNPOD_PORT" ]]; then
  echo "Set RUNPOD_HOST and RUNPOD_PORT from RunPod's 'SSH over exposed TCP' connection details." >&2
  echo "The ssh.runpod.io proxy is shell-only and does not support rsync/scp." >&2
  exit 2
fi

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 LOCAL_SHARD REMOTE_DATA_DIR" >&2
  echo "Example: $0 local_data/shards/davis2017_val /workspace/data/davis2017_val" >&2
  exit 2
fi

LOCAL_SHARD="$1"
REMOTE_DIR="$2"
RSYNC_SSH="ssh -o IdentitiesOnly=yes -i ${SSH_KEY} -p ${RUNPOD_PORT}"
rsync -avP --partial --progress \
  -e "$RSYNC_SSH" \
  "${LOCAL_SHARD%/}/" "${RUNPOD_HOST}:${REMOTE_DIR%/}/"
