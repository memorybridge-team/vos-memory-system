# Design review environment: repaired prep env + separate eval libraries. Nothing here changes the prep folder.
. /workspace/cmmt-large-prep-20261103/env.sh
export REVIEW_ROOT=/workspace/cmmt-design-review-20261104
export PYTHONPATH="$REVIEW_ROOT/code:$REVIEW_ROOT/pylibs:$PYTHONPATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
