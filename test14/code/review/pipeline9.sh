#!/usr/bin/env bash
# Third fine-tuning round: half the draws read the conditioning record alone, so it has to find the object without recents.
# Compares the context model against the depthwise model with context, which had the best correlation under MSE.
# J&F follows on MOSE; LVOS stays manual until MOSE has picked a model.
# Independent of any SSH or Claude session; training resumes from last.pt and finished cases are skipped on a rerun.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline9
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

E2E3=(--updates 12000 --batch 4 --lr 1e-4 --alone-share .5 --cond-share .8 --ridge-pointer
      --w-feature 10 --w-mask 5 --w-score 1 --w-logit .1 --w-dice 1 --lam .1 --lvos-share .25 --held-out 512)
T12=(--method e2e3_ctx_f=run:spatial/e2e3_ctx --method e2e3_dsc_f=run:spatial/e2e3_dsc)

train() {  # train TAG INIT
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=0
    retry $1 logs/pipeline9/$1.log python code/train_e2e.py --tag $1 --init $2 "${E2E3[@]}"
  )
}

flow() {  # flow GPU
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$1
    retry t12_mose_$1 logs/pipeline9/t12_mose_$1.log python code/more_eval.py --autocast --skip-small --shard $1 --shards 2 \
      --split dev --dataset MOSEv2 --limit 300 --tag t12_mose "${T12[@]}"
  )
}

log "pipeline9 started pid $$"
train e2e3_ctx e2e_ctx_last &
A=$!
train e2e3_dsc ds_context &
B=$!
wait $A $B
flow 0 &
C=$!
flow 1 &
D=$!
wait $C $D
log "pipeline9 finished"
date '+%F %T' > runs/reports/ALL_DONE9
