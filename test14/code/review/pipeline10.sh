#!/usr/bin/env bash
# LVOS for the best fine-tuned model so far, with and without a pinned record. The model is picked on MOSE; this checks long videos.
# Independent of any SSH or Claude session; finished cases are skipped on a rerun.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline10
log() { echo "$(date '+%F %T') $*" >> $R/logs/pipeline2/supervisor.log; }

retry() {  # retry NAME LOGFILE command...  stops after two clean passes or six attempts
  local name=$1 file=$2 clean=0
  shift 2
  for attempt in 1 2 3 4 5 6; do
    log "start $name attempt $attempt"
    if "$@" >> "$file" 2>&1; then
      clean=$((clean + 1))
      log "pass $name ($clean)"
      [ $clean -ge 2 ] && return 0
    else
      log "FAILED $name attempt $attempt (exit $?)"
    fi
    sleep 20
  done
  log "GAVE UP $name"
  return 1
}

CTX=run:spatial/e2e_ctx_last
T=(--method e2e_ctx_f=$CTX --method e2e_ctx_pin7=pin:7:$CTX)

flow() {
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$1
    retry t10_val_$1 logs/pipeline10/t10_val_$1.log python code/more_eval.py --autocast --skip-small --shard $1 --shards 2 --split val --tag t10_val "${T[@]}"
  )
}

log "pipeline10 started pid $$"
flow 0 &
A=$!
flow 1 &
B=$!
wait $A $B
log "pipeline10 finished"
date '+%F %T' > runs/reports/ALL_DONE10
