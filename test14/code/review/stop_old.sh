#!/usr/bin/env bash
# Stops pipeline5 and the tail of pipeline6 by exact PID, keeping the e2e_mlp training alive. Prints every PID it acts on.
cd /workspace/cmmt-design-review-20261104
for pid in 52855 52859 52860 52861 52862 56905 56908; do
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  case "$cmd" in
    "bash code/pipeline5.sh "*|"bash code/pipeline6.sh "*) kill -TERM $pid && echo "stopped $pid: $cmd";;
    *) echo "SKIPPED $pid: $cmd";;
  esac
done
for pid in 52869 52871; do
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  case "$cmd" in
    *"more_eval.py"*"--tag t7_mose"*) kill -CONT $pid; kill -TERM $pid && echo "stopped $pid: ${cmd:0:90}";;
    *) echo "SKIPPED $pid: $cmd";;
  esac
done
sed -i 's/^e2e_probe t9_mose /e2e_probe t9_mose t10_mose t11_mose /' runs/control/order
cat runs/control/order
nohup setsid bash code/pipeline8.sh < /dev/null > /dev/null 2>&1 &
echo started pipeline8
