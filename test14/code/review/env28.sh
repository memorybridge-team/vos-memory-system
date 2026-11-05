# J&F evaluation on torch 2.8: the prebuilt SAM2 _C.so leaks GPU memory under torch 2.11.
. /workspace/cmmt-large-prep-20261103/env.sh
export REVIEW_ROOT=/workspace/cmmt-design-review-20261104
SP=/workspace/j700qtcgrg/CMMT-official-isolated/official-20261005T012256KST/.venv/lib/python3.12/site-packages
export PYTHONPATH="$REVIEW_ROOT/code:/workspace/cmmt-large-prep-20261103/source/src:$CMMT_SAM2_ROOT:$SP:$REVIEW_ROOT/pylibs"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TQDM_DISABLE=1 REVIEW_GPU_FRACTION=0.45
