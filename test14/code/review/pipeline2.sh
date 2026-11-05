#!/usr/bin/env bash
# Second round of the design review. Runs on the server, independent of any SSH or Claude
# session and of pipeline.sh. Every stage is resumable: finished cases are skipped.
R=/workspace/cmmt-design-review-20261104
cd $R
mkdir -p logs/pipeline2 runs/reports
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

# Doc baselines and memory compositions, with Small's last visible mask.
T1=(--method target_ref=target --method direct=direct --method direct_nc=target,direct,target,direct
    --method ridge_nc=target,ridge,target,ridge --method legacy_nc=target,legacy,target,legacy
    --method orig=orig --method lastvis=lastvis --method olv=olv
    --method ridge_nc_anchor=anchorc:ridge --method ridge_anchor=anchor:ridge --method recent_ridge=recent:ridge)
# Does a better model fix full translation? What error size matters, and does its structure?
T2=(--method target_ref=target --method legacy=legacy --method ridge=ridge
    --method mlp=run:spatial/mlp --method transformer=run:spatial/transformer --method mlp_context=run:spatial/mlp_context
    --method ds3=run:spatial/ds3 --method ds_context=run:spatial/ds_context
    --method cond25=blend:0.25:legacy,target,target,target --method cond50=blend:0.5:legacy,target,target,target
    --method cond_noise=noise:1:legacy,target,target,target --method recent_noise=target,noise:1:legacy,target,target
    --method ridge_gain=rescale:1.1:ridge)
T3=(--method target_ref=target --method ceiling=run:spatial/ceiling --method ceiling_nc=target,run:spatial/ceiling,target,run:spatial/ceiling)
LIVE=(--method direct=direct --method direct_nc=target,direct,target,direct --method orig=orig --method lastvis=lastvis
      --method recent_ridge=recent:ridge --method olv=orig_lastvis)
# Pair-matched precision on a subset: how much does the bf16 continuation move the conclusions?
FP32=(--method target_ref=target --method legacy=legacy --method legacy_nc=target,legacy,target,legacy
      --method ridge_nc=target,ridge,target,ridge --method olv=olv)

flow_compositions() {
  local gpu=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    retry t1_val_$gpu logs/pipeline2/t1_val_$gpu.log python code/handoff_eval.py --autocast --split val --tag t1_val --shard $gpu --shards 2 "${T1[@]}"
    retry t1_mose_$gpu logs/pipeline2/t1_mose_$gpu.log python code/handoff_eval.py --autocast --split dev --dataset MOSEv2 --limit 300 --tag t1_mose --shard $gpu --shards 2 "${T1[@]}"
    retry pumavos2_$gpu logs/pipeline2/pumavos2_$gpu.log python code/unseen_more.py --dataset pumavos --tag pumavos2 --fractions 0.25,0.5,0.75 --shard $gpu --shards 2 "${LIVE[@]}"
    retry m3vos2_$gpu logs/pipeline2/m3vos2_$gpu.log python code/unseen_more.py --dataset m3vos --tag m3vos2 --max-videos 81 --fractions 0.5 --shard $gpu --shards 2 "${LIVE[@]}"
    retry fp32_val_$gpu logs/pipeline2/fp32_val_$gpu.log python code/handoff_eval.py --split val --limit 20 --tag fp32_val --shard $gpu --shards 2 "${FP32[@]}"
    retry fp32_mose_$gpu logs/pipeline2/fp32_mose_$gpu.log python code/handoff_eval.py --split dev --dataset MOSEv2 --limit 60 --tag fp32_mose --shard $gpu --shards 2 "${FP32[@]}"
  )
}

flow_models() {
  local gpu=$1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=$gpu
    retry t2_val_$gpu logs/pipeline2/t2_val_$gpu.log python code/handoff_eval.py --autocast --skip-small --split val --tag t2_val --shard $gpu --shards 2 "${T2[@]}"
    retry t2_mose_$gpu logs/pipeline2/t2_mose_$gpu.log python code/handoff_eval.py --autocast --skip-small --split dev --dataset MOSEv2 --limit 300 --tag t2_mose --shard $gpu --shards 2 "${T2[@]}"
  )
}

flow_ceiling() {
  (
    . $R/env.sh; export CUDA_VISIBLE_DEVICES=1
    [ -f runs/spatial/ceiling/result.json ] || retry train_ceiling logs/train/ceiling2.log python code/train_spatial.py --name ceiling
  )
  [ -f runs/spatial/ceiling/result.json ] || return 1
  (
    . $R/env28.sh; export CUDA_VISIBLE_DEVICES=1
    retry t3_val logs/pipeline2/t3_val.log python code/handoff_eval.py --autocast --skip-small --split val --tag t3_val "${T3[@]}"
    retry t3_mose logs/pipeline2/t3_mose.log python code/handoff_eval.py --autocast --skip-small --split dev --dataset MOSEv2 --limit 300 --tag t3_mose "${T3[@]}"
  )
}

reports() {
  (
    . $R/env.sh; export CUDA_VISIBLE_DEVICES=""
    for tag in sens_val sens_mose cand_val cand_mose t1_val t1_mose t2_val t2_mose t3_val t3_mose fp32_val fp32_mose; do
      [ -d runs/jf/$tag/cases ] && [ -n "$(ls runs/jf/$tag/cases)" ] && python code/summarize_jf.py $tag > runs/reports/$tag.txt 2>&1
    done
    for tag in pumavos m3vos pumavos2 m3vos2; do
      [ -d runs/unseen/$tag/cases ] && [ -n "$(ls runs/unseen/$tag/cases)" ] && python code/summarize_unseen.py $tag > runs/reports/$tag.txt 2>&1
    done
    python code/summarize_spatial.py > runs/reports/spatial.txt 2>&1
    date '+%F %T' > runs/reports/updated_at.txt
  )
}

log "pipeline2 started pid $$"
( while true; do reports; sleep 600; done ) &
REPORTER=$!

flow_compositions 0 &
A=$!
flow_compositions 1 &
B=$!
flow_models 0 &
C=$!
flow_models 1 &
D=$!
flow_ceiling &
E=$!
wait $A $B $C $D $E
while pgrep -f 'code/pipeline.sh' > /dev/null; do sleep 60; done
kill $REPORTER 2>/dev/null
reports
log "pipeline2 finished"
date '+%F %T' > runs/reports/ALL_DONE2
