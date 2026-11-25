#!/usr/bin/env bash
# Replaces the contrast round of pipeline5 and the J&F round of pipeline6.
# model.pt of a fine-tuned run froze on an early step, picked by a 256-item IoU where one item moves the score by .004,
# so J&F uses the weights of the last update. Fine-tuning restores contrast by itself, so only one gain row stays, for comparison.
# The held-out LVOS run stays manual until MOSE has picked a model.
# Independent of any SSH or Claude session; finished cases are skipped on a rerun.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline8
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
T10=(--method e2e_ctx_f=$CTX
     --method e2e_ctx_fc=$CTX,run:spatial/mlp_context,$CTX,run:spatial/mlp_context
     --method e2e_ctx_pin7=pin:7:$CTX
     --method m110_ctx=rescale:1.1:run:spatial/mlp_context)
T11=(--method e2e_mlp_f=run:spatial/e2e_mlp_last
     --method e2e2_ctx_f=run:spatial/e2e2_ctx_last)
E2E=(--updates 12000 --batch 4 --lr 1e-4 --w-feature 10 --w-mask 5 --w-score .5 --lam .1)

flow() {  # flow TAG GPU methods...
  local tag=$1 gpu=$2
  shift 2
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    retry ${tag}_$gpu logs/pipeline8/${tag}_$gpu.log python code/more_eval.py --autocast --skip-small --shard $gpu --shards 2 \
      --split dev --dataset MOSEv2 --limit 300 --tag $tag "$@"
  )
}

log "pipeline8 started pid $$"
flow t10_mose 0 "${T10[@]}" &
A=$!
flow t10_mose 1 "${T10[@]}" &
B=$!
while kill -0 56910 2>/dev/null; do sleep 20; done
(
  . $R/env28.sh; export CUDA_VISIBLE_DEVICES=0
  retry e2e_mlp_gpu0 logs/pipeline5/e2e_mlp.log python code/train_e2e.py --tag e2e_mlp --init mlp "${E2E[@]}"
)
while [ ! -f runs/spatial/e2e_mlp/result.json ] || [ ! -f runs/spatial/e2e2_ctx/result.json ]; do sleep 20; done
(
  . $R/env.sh; export CUDA_VISIBLE_DEVICES=""
  python code/use_last.py e2e_mlp e2e2_ctx >> logs/pipeline8/use_last.log 2>&1
)
flow t11_mose 0 "${T11[@]}" &
C=$!
flow t11_mose 1 "${T11[@]}" &
D=$!
wait $A $B $C $D
log "pipeline8 finished"
date '+%F %T' > runs/reports/ALL_DONE5
date '+%F %T' > runs/reports/ALL_DONE6
