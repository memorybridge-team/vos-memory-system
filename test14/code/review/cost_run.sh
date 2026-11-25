#!/usr/bin/env bash
# Pauses the experiments for a clean timing run, then restores the agreed limit whatever happens.
R=/workspace/cmmt-design-review-20261104
cd $R
trap 'echo 1 > runs/control/max_per_gpu' EXIT
echo 0 > runs/control/max_per_gpu
sleep 5
. ./env28.sh
export CUDA_VISIBLE_DEVICES=0 REVIEW_GPU_FRACTION=0.45
timeout 1500 python code/cost_eval.py > logs/pipeline3/cost.log 2>&1
