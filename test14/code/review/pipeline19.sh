#!/usr/bin/env bash
# The other two designs after the second round on the external datasets, which neither trained nor chose anything.
# The third round added nothing on M3-VOS, so the verdict on the three designs compares them at the second round there.
# Both shards run on GPU 0, which may be used heavily. Independent of any SSH or Claude session; finished cases are skipped on a rerun.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline19
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

U=(--method e2e_mlp_f=run:spatial/e2e_mlp_last --method e2e_tf_f=run:spatial/e2e_tf)

flow() {
  local shard=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=0
    retry m3vos5_$shard logs/pipeline19/m3vos5_$shard.log python code/unseen_more.py --dataset m3vos --tag m3vos5 --max-videos 81 --fractions 0.5 --shard $shard --shards 2 "${U[@]}"
    retry pumavos5_$shard logs/pipeline19/pumavos5_$shard.log python code/unseen_more.py --dataset pumavos --tag pumavos5 --fractions 0.25,0.5,0.75 --shard $shard --shards 2 "${U[@]}"
  )
}

log "pipeline19 started pid $$"
flow 0 &
A=$!
flow 1 &
B=$!
wait $A $B
log "pipeline19 finished"
date '+%F %T' > runs/reports/ALL_DONE19
