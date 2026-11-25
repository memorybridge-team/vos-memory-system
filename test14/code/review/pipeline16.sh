#!/usr/bin/env bash
# Final candidates, picked on the dev cases of MOSE and LVOS: e2e3_ctx, with a pinned record at +5 where pooled dev J&F is highest.
# LVOS val is the held-out test; PUMaVOS and M3-VOS are outside training. e2e_mlp is there for the verdict on the design.
# Independent of any SSH or Claude session; finished cases are skipped on a rerun.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline16
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

CTX=run:spatial/e2e3_ctx
T17=(--method e2e3_ctx_f=$CTX --method e2e3_ctx_pin5=pin:5:$CTX --method e2e_mlp_f=run:spatial/e2e_mlp_last)
U=(--method target_ref=target --method e2e3_ctx_f=$CTX --method e2e3_ctx_pin5=pin:5:$CTX)

flow() {
  local gpu=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    retry t17_val_$gpu logs/pipeline16/t17_val_$gpu.log python code/more_eval.py --autocast --skip-small --shard $gpu --shards 2 --split val --tag t17_val "${T17[@]}"
    retry m3vos4_$gpu logs/pipeline16/m3vos4_$gpu.log python code/unseen_more.py --dataset m3vos --tag m3vos4 --max-videos 81 --fractions 0.5 --shard $gpu --shards 2 "${U[@]}"
    retry pumavos4_$gpu logs/pipeline16/pumavos4_$gpu.log python code/unseen_more.py --dataset pumavos --tag pumavos4 --fractions 0.25,0.5,0.75 --shard $gpu --shards 2 "${U[@]}"
  )
}

log "pipeline16 started pid $$"
flow 0 &
A=$!
flow 1 &
B=$!
wait $A $B
log "pipeline16 finished"
date '+%F %T' > runs/reports/ALL_DONE16
