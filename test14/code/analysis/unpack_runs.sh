#!/usr/bin/env bash
# Rebuilds the runs/ layout build_tables.py reads, from the archives of test14. Usage: unpack_runs.sh <empty folder>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$1"
mkdir -p "$OUT/jf" "$OUT/unseen" "$OUT/analysis" "$OUT/cost"
for group in jf unseen; do
  for archive in "$ROOT/results/raw/$group"/*.tar.gz; do
    tar -xzf "$archive" -C "$OUT/$group"
  done
done
cp "$ROOT/results/analysis/dup_scan.json" "$OUT/analysis/"
cp "$ROOT/results/analysis/cost.json" "$OUT/cost/"
cp -R "$ROOT/weights/spatial" "$ROOT/weights/pointer" "$OUT/"
echo "unpacked into $OUT"
