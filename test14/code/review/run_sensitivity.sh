#!/usr/bin/env bash
# usage: run_sensitivity.sh GPU SHARD
. /workspace/cmmt-design-review-20261104/env.sh
cd $REVIEW_ROOT
export CUDA_VISIBLE_DEVICES=$1
M=(--method target_ref=target --method legacy=legacy --method ridge=ridge
   --method legacy_sp_only=legacy,legacy,target,target --method legacy_ptr_only=target,target,legacy,legacy
   --method legacy_cond_oracle=target,legacy,target,legacy --method legacy_recent_oracle=legacy,target,legacy,target
   --method legacy_meanptr=legacy,legacy,meanptr,meanptr
   --method ridge_sp_only=ridge,ridge,target,target --method ridge_ptr_only=target,target,ridge,ridge)
python code/jf_eval.py --split val --tag sens_val --shard $2 --shards 2 "${M[@]}" > logs/jf/sens_val_$2.log 2>&1
python code/jf_eval.py --split dev --dataset MOSEv2 --limit 300 --tag sens_mose --shard $2 --shards 2 "${M[@]}" > logs/jf/sens_mose_$2.log 2>&1
