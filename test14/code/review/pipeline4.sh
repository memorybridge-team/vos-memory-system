#!/usr/bin/env bash
# Fourth round: raising full-translation accuracy. No past RGB in any method here.
# Independent of any SSH or Claude session. Finished cases are skipped on a rerun.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline4 runs/reports
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

# Contrast gain on the translated state, and Base+'s own record replacing the translated conditioning record.
T5=(--method target_ref=target
    --method g110=rescale:1.1:legacy --method g120=rescale:1.2:legacy --method g130=rescale:1.3:legacy --method g150=rescale:1.5:legacy
    --method swap1=swap:1:legacy --method swap7=swap:7:legacy --method swap30=swap:30:legacy --method pin7=pin:7:legacy
    --method swap7g=swap:7:rescale:1.2:legacy
    --method swap7_ridge=swap:7:ridge --method swap7_mlp=swap:7:run:spatial/mlp --method swap7_ctx=swap:7:run:spatial/mlp_context)
# Which part of the conditioning-record error does the damage?
C=target,target,target
T6=(--method target_ref=target --method cond_err=legacy,$C
    --method err_near=anatomy:near:legacy,$C --method err_far=anatomy:far:legacy,$C
    --method err_low=anatomy:low:legacy,$C --method err_high=anatomy:high:legacy,$C
    --method err_shrink=anatomy:shrink:legacy,$C --method err_rest=anatomy:rest:legacy,$C)

flow() {
  local gpu=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    M=(--autocast --skip-small --shard $gpu --shards 2)
    retry t5_mose_$gpu logs/pipeline4/t5_mose_$gpu.log python code/more_eval.py "${M[@]}" --split dev --dataset MOSEv2 --limit 300 --tag t5_mose "${T5[@]}"
    retry t6_mose_$gpu logs/pipeline4/t6_mose_$gpu.log python code/more_eval.py "${M[@]}" --split dev --dataset MOSEv2 --limit 300 --tag t6_mose "${T6[@]}"
    retry t5_val_$gpu logs/pipeline4/t5_val_$gpu.log python code/more_eval.py "${M[@]}" --split val --tag t5_val "${T5[@]}"
    retry t6_val_$gpu logs/pipeline4/t6_val_$gpu.log python code/more_eval.py "${M[@]}" --split val --tag t6_val "${T6[@]}"
  )
}

log "pipeline4 started pid $$"
flow 0 &
A=$!
flow 1 &
B=$!
wait $A $B
log "pipeline4 finished"
date '+%F %T' > runs/reports/ALL_DONE4
