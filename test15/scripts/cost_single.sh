#!/usr/bin/env bash
# One method per process, one process at a time, alone on GPU 0.
cd /workspace/memory-resume-20261120 && . ./env.sh
for m in resume paper conv_strip_vm restart_p0 restart_mask restart_p0_mask; do
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader > checks/single/$m.gpu_before.txt
  CUDA_VISIBLE_DEVICES=0 python3 code/cost_single.py --method $m --out checks/single/$m.json > logs/cost_single_$m.log 2>&1
done
echo done > checks/single/DONE
