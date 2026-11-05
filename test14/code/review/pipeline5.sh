#!/usr/bin/env bash
# Fifth round: contrast calibration across models, and translators fine-tuned on Base+'s downstream output.
# No past RGB in any evaluated method. Independent of any SSH or Claude session; finished cases are skipped.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline5 runs/reports
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

# Energy-matching gains come from dev record statistics, not from J&F (code/band_stats.py).
T7=(--method legacy=legacy
    --method m110_mlp=rescale:1.1:run:spatial/mlp
    --method m110_ctx=rescale:1.1:run:spatial/mlp_context
    --method m110_ds3=rescale:1.1:run:spatial/ds3
    --method m110_dsc=rescale:1.1:run:spatial/ds_context
    --method gc110=rescale:1.1:legacy,legacy,legacy,legacy
    --method gr110=legacy,rescale:1.1:legacy,legacy,legacy
    --method band_em=band:1.035:1.15:legacy
    --method band_h=band:1:1.15:legacy
    --method cal_leg=cal:legacy:legacy
    --method cal_ctx=cal:mlp_context:run:spatial/mlp_context)

T8=(--method e2e_mlp=run:spatial/e2e_mlp --method e2e_ctx=run:spatial/e2e_ctx
    --method e2e_mlp_c=run:spatial/e2e_mlp,run:spatial/mlp,run:spatial/e2e_mlp,run:spatial/mlp
    --method e2e_ctx_c=run:spatial/e2e_ctx,run:spatial/mlp_context,run:spatial/e2e_ctx,run:spatial/mlp_context)

E2E=(--updates 12000 --batch 4 --lr 1e-4 --w-feature 10 --w-mask 5 --w-score .5 --lam .1)

flow() {
  local gpu=$1 model=$2
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    M=(--autocast --skip-small --shard $gpu --shards 2)
    retry e2e_$model logs/pipeline5/e2e_$model.log python code/train_e2e.py --tag e2e_$3 --init $model "${E2E[@]}" &
    retry t7_mose_$gpu logs/pipeline5/t7_mose_$gpu.log python code/more_eval.py "${M[@]}" --split dev --dataset MOSEv2 --limit 300 --tag t7_mose "${T7[@]}"
    retry t7_val_$gpu logs/pipeline5/t7_val_$gpu.log python code/more_eval.py "${M[@]}" --split val --tag t7_val "${T7[@]}"
    wait
    while [ ! -f runs/spatial/e2e_mlp/result.json ] || [ ! -f runs/spatial/e2e_ctx/result.json ]; do sleep 60; done
    retry t8_mose_$gpu logs/pipeline5/t8_mose_$gpu.log python code/more_eval.py "${M[@]}" --split dev --dataset MOSEv2 --limit 300 --tag t8_mose "${T8[@]}"
    retry t8_val_$gpu logs/pipeline5/t8_val_$gpu.log python code/more_eval.py "${M[@]}" --split val --tag t8_val "${T8[@]}"
  )
}

log "pipeline5 started pid $$"
flow 1 mlp mlp &
A=$!
flow 0 mlp_context ctx &
B=$!
wait $A $B
log "pipeline5 finished"
date '+%F %T' > runs/reports/ALL_DONE5
