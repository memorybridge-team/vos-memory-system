# memory-resume-20261120: system Python 3.12 + torch 2.8 from the official venv, the design review's evaluator code,
# experiments_v5 (paper / strip: tokens). Same module path order as cmmt-design-review-20261104/env28.sh, whose
# interpreter folder (cmmt-large-prep-20261103/env/) is gone. $E/pylibs links SAM2's config dependencies (hydra,
# omegaconf, ...) from test10-affine-comparison/eval-deps, as residual-vs-mlp-20261116 did.
E=/workspace/memory-resume-20261120
W=/workspace/linear-transfer-large-20261108
export REVIEW_ROOT=/workspace/cmmt-design-review-20261104
export CMMT_SAM2_ROOT=/workspace/j700qtcgrg/CMMT-task07-repair-20261003/sam2
SP=/workspace/j700qtcgrg/CMMT-official-isolated/official-20261005T012256KST/.venv/lib/python3.12/site-packages
export PYTHONPATH="$E/code:$W/experiments_v5:$REVIEW_ROOT/code:/workspace/cmmt-large-prep-20261103/source/src:$CMMT_SAM2_ROOT:$SP:$E/pylibs:$REVIEW_ROOT/pylibs"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TQDM_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export LIN_RUN=$E LIN_TARGET=base LIN_MAPPERS=$W/vos/mappers_fp32.pt LIN_GAINS=$W/vos/variance_gain.pt \
  LIN_STRIP_MAPPERS=$W/runs/sb-linear-20261113/mappers_strip_fp32.pt LIN_STRIP_GAINS=$W/runs/sb-linear-20261113/variance_gain_strip.pt
