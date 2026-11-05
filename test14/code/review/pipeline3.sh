#!/usr/bin/env bash
# Third round: doc baselines not measured yet, and anchors that do not trust Small's mask.
# Independent of any SSH or Claude session. Finished cases are skipped on a rerun.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline3 runs/reports
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

T4=(--method target_ref=target --method mm=mm --method mm_nc=target,mm,target,mm
    --method replay4=replay:4 --method replay8=replay:8 --method replay16=replay:16
    --method olv_soft=olv_soft --method pinb=pinb:ridge --method pinp=pinp)

flow() {
  local gpu=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    retry t4_val_$gpu logs/pipeline3/t4_val_$gpu.log python code/more_eval.py --autocast --split val --tag t4_val --shard $gpu --shards 2 "${T4[@]}"
    retry t4_mose_$gpu logs/pipeline3/t4_mose_$gpu.log python code/more_eval.py --autocast --split dev --dataset MOSEv2 --limit 300 --tag t4_mose --shard $gpu --shards 2 "${T4[@]}"
  )
}

while [ ! -f runs/moment/moment.pt ]; do sleep 30; done
log "pipeline3 started pid $$"
flow 0 &
A=$!
flow 1 &
B=$!
wait $A $B
log "pipeline3 finished"
date '+%F %T' > runs/reports/ALL_DONE3
