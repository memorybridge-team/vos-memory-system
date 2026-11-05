#!/usr/bin/env bash
# e2e_mlp runs on GPU 0 next to e2e_ctx, because GPU 1 is the loud one and GPU 0 may be used heavily.
# J&F of the fine-tuned models starts as soon as both finish, without waiting for the contrast round in pipeline5.
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

E2E=(--updates 12000 --batch 4 --lr 1e-4 --w-feature 10 --w-mask 5 --w-score .5 --lam .1)
T8=(--method e2e_mlp=run:spatial/e2e_mlp --method e2e_ctx=run:spatial/e2e_ctx
    --method e2e_mlp_c=run:spatial/e2e_mlp,run:spatial/mlp,run:spatial/e2e_mlp,run:spatial/mlp
    --method e2e_ctx_c=run:spatial/e2e_ctx,run:spatial/mlp_context,run:spatial/e2e_ctx,run:spatial/mlp_context)

flow() {
  local gpu=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    M=(--autocast --skip-small --shard $gpu --shards 2)
    retry t8_mose_$gpu logs/pipeline5/t8_mose_$gpu.log python code/more_eval.py "${M[@]}" --split dev --dataset MOSEv2 --limit 300 --tag t8_mose "${T8[@]}"
    retry t8_val_$gpu logs/pipeline5/t8_val_$gpu.log python code/more_eval.py "${M[@]}" --split val --tag t8_val "${T8[@]}"
  )
}

log "pipeline6 started pid $$"
(
  . $R/env28.sh; export CUDA_VISIBLE_DEVICES=0
  retry e2e_mlp_gpu0 logs/pipeline5/e2e_mlp.log python code/train_e2e.py --tag e2e_mlp --init mlp "${E2E[@]}"
)
while [ ! -f runs/spatial/e2e_mlp/result.json ] || [ ! -f runs/spatial/e2e_ctx/result.json ]; do sleep 30; done
flow 0 &
A=$!
flow 1 &
B=$!
wait $A $B
log "pipeline6 finished"
date '+%F %T' > runs/reports/ALL_DONE6
