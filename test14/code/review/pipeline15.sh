#!/usr/bin/env bash
# The third design under the same fine-tuning as e2e_mlp and e2e_ctx, so the verdict on its structure does not rest on MSE alone.
# Independent of any SSH or Claude session; training resumes from last.pt and finished cases are skipped on a rerun.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline15
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
T18=(--method e2e_tf_f=run:spatial/e2e_tf)

flow() {
  local gpu=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    M=(--autocast --skip-small --shard $gpu --shards 2)
    retry t18_mose_$gpu logs/pipeline15/t18_mose_$gpu.log python code/more_eval.py "${M[@]}" --split dev --dataset MOSEv2 --limit 300 --tag t18_mose "${T18[@]}"
    retry t18_val_$gpu logs/pipeline15/t18_val_$gpu.log python code/more_eval.py "${M[@]}" --split val --tag t18_val "${T18[@]}"
  )
}

log "pipeline15 started pid $$"
(
  . $R/env28.sh; export CUDA_VISIBLE_DEVICES=0
  retry e2e_tf logs/pipeline15/e2e_tf.log python code/train_e2e.py --tag e2e_tf --init transformer "${E2E[@]}"
)
flow 0 &
A=$!
flow 1 &
B=$!
wait $A $B
log "pipeline15 finished"
date '+%F %T' > runs/reports/ALL_DONE15
