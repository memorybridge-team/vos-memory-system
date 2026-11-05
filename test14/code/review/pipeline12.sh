#!/usr/bin/env bash
# Fine-tuned translator and the pinned record on datasets outside training: PUMaVOS and M3-VOS, live states as in unseen_eval.
# Independent of any SSH or Claude session; finished cases are skipped on a rerun.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline12
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
U=(--method target_ref=target --method legacy=legacy --method mlp_context=run:spatial/mlp_context
   --method e2e_ctx_f=$CTX --method e2e_ctx_pin7=pin:7:$CTX)

flow() {
  local gpu=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    retry pumavos3_$gpu logs/pipeline12/pumavos3_$gpu.log python code/unseen_more.py --dataset pumavos --tag pumavos3 --fractions 0.25,0.5,0.75 --shard $gpu --shards 2 "${U[@]}"
    retry m3vos3_$gpu logs/pipeline12/m3vos3_$gpu.log python code/unseen_more.py --dataset m3vos --tag m3vos3 --max-videos 81 --fractions 0.5 --shard $gpu --shards 2 "${U[@]}"
  )
}

log "pipeline12 started pid $$"
flow 0 &
A=$!
flow 1 &
B=$!
wait $A $B
log "pipeline12 finished"
date '+%F %T' > runs/reports/ALL_DONE12
