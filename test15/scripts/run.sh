#!/usr/bin/env bash
# usage: run.sh GPU DATASET TAG SHARD SHARDS [extra args, e.g. --names chair,dog_tail]   (memory-resume-20261120)
# Cases as in every earlier table: PUMaVOS 78 cases (switch 25/50/75%), M3-VOS 110 cases (--max-videos 81, switch 50%).
# Finished case files are skipped, so a rerun resumes.
E=/workspace/memory-resume-20261120
. $E/env.sh
export CUDA_VISIBLE_DEVICES=$1 REVIEW_GPU_FRACTION=${FRACTION:-0.5}
D=$2; TAG=$3; SHARD=$4; SHARDS=$5; shift 5
if [ "$D" = pumavos ]; then F=(--fractions 0.25,0.5,0.75); else F=(--max-videos 81 --fractions 0.5); fi
C=lin:S2B:conv3_mean:pointer_mean:shared
M=(--method resume=resume --method restart_p0=orig --method restart_mask=selfmask --method restart_p0_mask=orig_selfmask
   --method olv=orig_lastvis --method lastvis=lastvis --method paper=paper --method conv_strip_vm=strip:$C:vm)
mkdir -p $E/logs
cd $E && exec python3 code/resume_eval.py --dataset "$D" --tag "$TAG" --shard "$SHARD" --shards "$SHARDS" "${F[@]}" "${M[@]}" "$@"
