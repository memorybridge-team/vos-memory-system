#!/usr/bin/env bash
# How the frame of the pinned record changes J&F, on MOSE only, with the best fine-tuned model so far.
# Independent of any SSH or Claude session; finished cases are skipped on a rerun.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline11
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
T=(--method e2e_ctx_pin3=pin:3:$CTX --method e2e_ctx_pin15=pin:15:$CTX --method e2e_ctx_pin30=pin:30:$CTX)
log "pipeline11 started pid $$"
. $R/env28.sh
export CUDA_VISIBLE_DEVICES=0
retry t13_mose logs/pipeline11/t13_mose.log python code/more_eval.py --autocast --skip-small --split dev --dataset MOSEv2 --limit 300 --tag t13_mose "${T[@]}"
log "pipeline11 finished"
date '+%F %T' > runs/reports/ALL_DONE11
