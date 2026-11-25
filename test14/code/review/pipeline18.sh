#!/usr/bin/env bash
# J&F on cases the models trained on, for a direct reading of how much the fine-tuned weights depend on their training videos.
# fit and dev are random video subsets of one pool, so the gap to Base+ on fit against the same gap on dev is the effect of exposure.
# mlp_context is the same design before fine-tuning. Both shards run on GPU 0, which may be used heavily.
# Independent of any SSH or Claude session; finished cases are skipped on a rerun.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline18
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

T20=(--method target_ref=target --method mlp_context=run:spatial/mlp_context --method e2e3_ctx_f=run:spatial/e2e3_ctx)

flow() {
  local shard=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=0
    M=(--autocast --skip-small --shard $shard --shards 2)
    retry t20_mose_$shard logs/pipeline18/t20_mose_$shard.log python code/more_eval.py "${M[@]}" --split fit --dataset MOSEv2 --limit 300 --tag t20_mose "${T20[@]}"
    retry t20_lvos_$shard logs/pipeline18/t20_lvos_$shard.log python code/more_eval.py "${M[@]}" --split fit --dataset 'LVOS v2' --limit 73 --tag t20_lvos "${T20[@]}"
  )
}

log "pipeline18 started pid $$"
# GPU 0 has room for one more process now: the first shard waits for the proxy, the second for m3vos3 or the first trainings to free memory.
( while [ ! -f runs/analysis/gap_proxy.json ]; do sleep 30; done; flow 0 ) &
A=$!
( while pgrep -f 'tag m3vos3' > /dev/null && [ ! -f runs/spatial/e2e3_mlp/result.json ]; do sleep 30; done; flow 1 ) &
B=$!
wait $A $B
log "pipeline18 finished"
date '+%F %T' > runs/reports/ALL_DONE18
