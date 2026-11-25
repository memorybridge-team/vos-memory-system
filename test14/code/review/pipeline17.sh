#!/usr/bin/env bash
# residual_mlp and transformer under the third fine-tuning round, the recipe that produced e2e3_ctx, so the three designs are compared on equal terms.
# The architecture is chosen on dev (MOSE and LVOS). LVOS val is reported for comparison and does not choose anything.
# Independent of any SSH or Claude session; training resumes from last.pt and finished cases are skipped on a rerun.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline17
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

E3=(--updates 12000 --batch 4 --lr 1e-4 --alone-share .5 --cond-share .8 --ridge-pointer --w-feature 10 --w-mask 5 --w-score 1 --w-logit .1 --w-dice 1 --lam .1 --lvos-share .25 --held-out 512)
T19=(--method e2e3_mlp_f=run:spatial/e2e3_mlp --method e2e3_tf_f=run:spatial/e2e3_tf)

flow() {
  local gpu=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    M=(--autocast --skip-small --shard $gpu --shards 2)
    retry t19_mose_$gpu logs/pipeline17/t19_mose_$gpu.log python code/more_eval.py "${M[@]}" --split dev --dataset MOSEv2 --limit 300 --tag t19_mose "${T19[@]}"
    retry t19_lvd_$gpu logs/pipeline17/t19_lvd_$gpu.log python code/more_eval.py "${M[@]}" --split dev --dataset 'LVOS v2' --limit 120 --tag t19_lvd "${T19[@]}"
    retry t19_val_$gpu logs/pipeline17/t19_val_$gpu.log python code/more_eval.py "${M[@]}" --split val --tag t19_val "${T19[@]}"
  )
}

log "pipeline17 started pid $$"
(
  . $R/env28.sh; export CUDA_VISIBLE_DEVICES=0
  retry e2e3_mlp logs/pipeline17/e2e3_mlp.log python code/train_e2e.py --tag e2e3_mlp --init e2e_mlp_last "${E3[@]}"
) &
A=$!
(
  while [ ! -f runs/spatial/e2e_tf/result.json ]; do sleep 30; done
  . $R/env28.sh; export CUDA_VISIBLE_DEVICES=0
  retry e2e3_tf logs/pipeline17/e2e3_tf.log python code/train_e2e.py --tag e2e3_tf --init e2e_tf "${E3[@]}"
) &
B=$!
wait $A $B
flow 0 &
A=$!
flow 1 &
B=$!
wait $A $B
log "pipeline17 finished"
date '+%F %T' > runs/reports/ALL_DONE17
