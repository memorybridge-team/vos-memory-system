#!/usr/bin/env bash
# Drops the candidate round of pipeline.sh, which fine-tuning has made obsolete, to free GPU memory. Exact PIDs, each checked first.
cd /workspace/cmmt-design-review-20261104
for pid in $(pgrep -x bash); do
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  case "$cmd" in
    "bash /workspace/cmmt-design-review-20261104/code/pipeline.sh "*) kill -TERM $pid && echo "stopped $pid: $cmd";;
  esac
done
for pid in 22239 26090; do
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  case "$cmd" in
    *"jf_eval.py"*"--tag cand_"*) kill -CONT $pid; kill -TERM $pid && echo "stopped $pid: ${cmd:0:80}";;
    *) echo "SKIPPED $pid: ${cmd:0:80}";;
  esac
done
date '+%F %T' > runs/reports/ALL_DONE
sed -i "s/ t12_mose t13_mose m3vos3 pumavos3 / t12_mose t13_mose t15_mose m3vos3 pumavos3 t14_lvd /" runs/control/order
cat runs/control/order
nohup setsid bash code/pipeline13.sh < /dev/null > /dev/null 2>&1 &
echo started pipeline13
