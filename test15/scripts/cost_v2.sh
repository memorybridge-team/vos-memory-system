#!/usr/bin/env bash
# Protocol v2 (DESIGN-cost-v2.md). usage: ./cost_v2.sh [--smoke]
# One method per process, one process at a time, alone on GPU 0. Rerunning resumes (finished switches are skipped).
cd /workspace/memory-resume-20261120 && . ./env.sh
if [ "$1" = --smoke ]; then D=checks/v2_smoke; X=--smoke; else D=checks/v2; X=; fi
mkdir -p $D logs
for m in resume paper conv_strip_vm restart_p0 restart_mask restart_p0_mask olv lastvis; do
  if [ -f $D/$m.json ] && grep -q '"complete": true' $D/$m.json; then continue; fi
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader > $D/$m.gpu_before.txt
  CUDA_VISIBLE_DEVICES=0 python3 code/cost_v2.py --method $m --out $D/$m.json $X >> logs/$(basename $D)_$m.log 2>&1 || echo "FAILED $m $(date)" >> $D/FAILED
done
echo "done $(date)" > $D/DONE
