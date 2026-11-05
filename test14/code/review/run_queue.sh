#!/usr/bin/env bash
# usage: run_queue.sh GPU name [name...]
. /workspace/cmmt-design-review-20261104/env.sh
cd $REVIEW_ROOT
export CUDA_VISIBLE_DEVICES=$1
shift
for spec in "$@"; do
  name=${spec%%:*}
  python code/train_spatial.py --name $name > logs/train/$name.log 2>&1
done
