#!/usr/bin/env bash
# The rest of candidate selection on the LVOS part of dev, once the third fine-tuning round is done.
# Independent of any SSH or Claude session; finished cases are skipped on a rerun.
R=/workspace/cmmt-design-review-20261104
cd $R
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
T16=(--method e2e_ctx_pin5=pin:5:$CTX --method e2e_mlp_f=run:spatial/e2e_mlp_last
     --method e2e3_ctx_f=run:spatial/e2e3_ctx --method e2e3_dsc_f=run:spatial/e2e3_dsc)

flow() {
  local gpu=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    retry t16_lvd_$gpu logs/pipeline13/t16_lvd_$gpu.log python code/more_eval.py --autocast --skip-small --shard $gpu --shards 2 --split dev --dataset 'LVOS v2' --limit 120 --tag t16_lvd "${T16[@]}"
  )
}

log "pipeline14 started pid $$"
while [ ! -f runs/spatial/e2e3_ctx/result.json ] || [ ! -f runs/spatial/e2e3_dsc/result.json ]; do sleep 30; done
flow 0 &
A=$!
flow 1 &
B=$!
wait $A $B
log "pipeline14 finished"
date '+%F %T' > runs/reports/ALL_DONE14
