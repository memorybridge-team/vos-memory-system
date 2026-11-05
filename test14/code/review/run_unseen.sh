#!/usr/bin/env bash
# usage: run_unseen.sh GPU SHARD
. /workspace/cmmt-design-review-20261104/env.sh
cd $REVIEW_ROOT
export CUDA_VISIBLE_DEVICES=$1
M=(--method legacy=legacy --method ridge=ridge
   --method legacy_nc=target,legacy,target,legacy --method ridge_nc=target,ridge,target,ridge
   --method ridge_anchor=anchor:ridge --method ridge_nc_anchor=anchorc:ridge --method olv=orig_lastvis)
python code/unseen_eval.py --dataset pumavos --tag pumavos --fractions 0.25,0.5,0.75 --shard $2 --shards 2 "${M[@]}" > logs/unseen/pumavos_$2.log 2>&1
python code/unseen_eval.py --dataset m3vos --tag m3vos --max-videos 81 --fractions 0.5 --shard $2 --shards 2 "${M[@]}" > logs/unseen/m3vos_$2.log 2>&1
