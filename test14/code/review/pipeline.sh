#!/usr/bin/env bash
# Runs on the server, independent of any SSH or Claude session.
# Every stage is resumable: finished cases are skipped, so a rerun only does what is missing.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline runs/reports
log() { echo "$(date '+%F %T') $*" >> $R/logs/pipeline/supervisor.log; }

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

SENS=(--method target_ref=target --method legacy=legacy --method ridge=ridge
      --method legacy_sp_only=legacy,legacy,target,target --method legacy_ptr_only=target,target,legacy,legacy
      --method legacy_cond_oracle=target,legacy,target,legacy --method legacy_recent_oracle=legacy,target,legacy,target
      --method legacy_meanptr=legacy,legacy,meanptr,meanptr
      --method ridge_sp_only=ridge,ridge,target,target --method ridge_ptr_only=target,target,ridge,ridge)
UNSEEN=(--method legacy=legacy --method ridge=ridge
        --method legacy_nc=target,legacy,target,legacy --method ridge_nc=target,ridge,target,ridge
        --method ridge_anchor=anchor:ridge --method ridge_nc_anchor=anchorc:ridge --method olv=orig_lastvis)

flow_sens() {
  local gpu=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    retry sens_val_$gpu logs/pipeline/sens_val_$gpu.log python code/jf_eval.py --split val --tag sens_val --shard $gpu --shards 2 "${SENS[@]}"
    retry sens_mose_$gpu logs/pipeline/sens_mose_$gpu.log python code/jf_eval.py --split dev --dataset MOSEv2 --limit 300 --tag sens_mose --shard $gpu --shards 2 "${SENS[@]}"
  )
}

flow_unseen() {
  local gpu=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    retry pumavos_$gpu logs/pipeline/pumavos_$gpu.log python code/unseen_eval.py --dataset pumavos --tag pumavos --fractions 0.25,0.5,0.75 --shard $gpu --shards 2 "${UNSEEN[@]}"
    retry m3vos_$gpu logs/pipeline/m3vos_$gpu.log python code/unseen_eval.py --dataset m3vos --tag m3vos --max-videos 81 --fractions 0.5 --shard $gpu --shards 2 "${UNSEEN[@]}"
  )
}

flow_candidates() {
  local gpu=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    local methods=(--method target_ref=target --method ridge_nc=target,ridge,target,ridge --method legacy_nc=target,legacy,target,legacy)
    for tag in mlp ds ds2 ds3 mlp_context transformer conv3mlp; do
      [ -f runs/spatial/$tag/result.json ] && methods+=(--method ${tag}_nc=target,run:spatial/$tag,target,run:spatial/$tag)
    done
    retry cand_val_$gpu logs/pipeline/cand_val_$gpu.log python code/jf_eval.py --split val --tag cand_val --shard $gpu --shards 2 "${methods[@]}"
    retry cand_mose_$gpu logs/pipeline/cand_mose_$gpu.log python code/jf_eval.py --split dev --dataset MOSEv2 --limit 300 --tag cand_mose --shard $gpu --shards 2 "${methods[@]}"
  )
}

reports() {
  (
    . $R/env.sh; export CUDA_VISIBLE_DEVICES=""
    for tag in sens_val sens_mose cand_val cand_mose; do
      [ -d runs/jf/$tag/cases ] && [ -n "$(ls runs/jf/$tag/cases)" ] && python code/summarize_jf.py $tag > runs/reports/$tag.txt 2>&1
    done
    for tag in pumavos m3vos; do
      [ -d runs/unseen/$tag/cases ] && [ -n "$(ls runs/unseen/$tag/cases)" ] && python code/summarize_unseen.py $tag > runs/reports/$tag.txt 2>&1
    done
    python code/summarize_spatial.py > runs/reports/spatial.txt 2>&1
    date '+%F %T' > runs/reports/updated_at.txt
  )
}

log "pipeline started pid $$"
( while true; do reports; sleep 600; done ) &
REPORTER=$!

flow_sens 0 &
A=$!
flow_sens 1 &
B=$!
flow_unseen 0 &
C=$!
flow_unseen 1 &
D=$!
wait $A $B
log "sensitivity flows finished"

# Spatial training: wait for the running queue, then retry anything that did not finish.
while pgrep -f 'code/run_queue.sh' > /dev/null; do sleep 60; done
(
  . $R/env.sh
  for name in affine mlp ds mlp_context ds2 conv3mlp mlp512 transformer transformer_ridge conv3lin ds3 ds_context mlp2; do
    [ -f runs/spatial/$name/result.json ] && continue
    CUDA_VISIBLE_DEVICES=1 retry train_$name logs/train/$name.log python code/train_spatial.py --name $name
  done
)
log "training finished"
flow_candidates 0 &
E=$!
flow_candidates 1 &
F=$!
wait $C $D $E $F
kill $REPORTER 2>/dev/null
reports
log "pipeline finished"
date '+%F %T' > runs/reports/ALL_DONE
