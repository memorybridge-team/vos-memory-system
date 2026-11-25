#!/usr/bin/env bash
# Earlier pinned records on MOSE, then candidate selection on the LVOS part of dev, so the LVOS val cases stay out of model choice.
# Independent of any SSH or Claude session; finished cases are skipped on a rerun.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline13
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
T15=(--method e2e_ctx_pin1=pin:1:$CTX --method e2e_ctx_pin2=pin:2:$CTX --method e2e_ctx_pin5=pin:5:$CTX)
T14=(--method target_ref=target --method e2e_ctx_f=$CTX --method e2e2_ctx_f=run:spatial/e2e2_ctx_last
     --method e2e_ctx_pin1=pin:1:$CTX --method e2e_ctx_pin3=pin:3:$CTX --method e2e_ctx_pin7=pin:7:$CTX)

flow() {
  local gpu=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    M=(--autocast --skip-small --shard $gpu --shards 2 --split dev)
    retry t15_mose_$gpu logs/pipeline13/t15_mose_$gpu.log python code/more_eval.py "${M[@]}" --dataset MOSEv2 --limit 300 --tag t15_mose "${T15[@]}"
    retry t14_lvd_$gpu logs/pipeline13/t14_lvd_$gpu.log python code/more_eval.py "${M[@]}" --dataset 'LVOS v2' --limit 120 --tag t14_lvd "${T14[@]}"
  )
}

log "pipeline13 started pid $$"
flow 0 &
A=$!
flow 1 &
B=$!
wait $A $B
log "pipeline13 finished"
date '+%F %T' > runs/reports/ALL_DONE13
