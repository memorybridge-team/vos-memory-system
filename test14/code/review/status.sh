#!/usr/bin/env bash
# One-screen progress view; safe to run at any time.
R=/workspace/cmmt-design-review-20261104
cd $R
count() { ls "$1" 2>/dev/null | wc -l; }
date '+%F %T'
echo "round 1: $(pgrep -f 'code/pipeline.sh' | wc -l) process(es) $( [ -f runs/reports/ALL_DONE ] && echo DONE )   round 2: $(pgrep -f 'code/pipeline2.sh' | wc -l) process(es) $( [ -f runs/reports/ALL_DONE2 ] && echo DONE )"
echo "[1] LVOS sens $(count runs/jf/sens_val/cases)/140   MOSE sens $(count runs/jf/sens_mose/cases)/300   PUMaVOS $(count runs/unseen/pumavos/cases)/26   M3-VOS $(count runs/unseen/m3vos/cases)/110"
echo "[1] candidates LVOS $(count runs/jf/cand_val/cases)/140   MOSE $(count runs/jf/cand_mose/cases)/300   spatial $(ls runs/spatial/*/result.json 2>/dev/null | wc -l)/14"
echo "[2] compositions LVOS $(count runs/jf/t1_val/cases)/140   MOSE $(count runs/jf/t1_mose/cases)/300   PUMaVOS $(count runs/unseen/pumavos2/cases)/26   M3-VOS $(count runs/unseen/m3vos2/cases)/110"
echo "[2] full translation LVOS $(count runs/jf/t2_val/cases)/140   MOSE $(count runs/jf/t2_mose/cases)/300"
echo "[2] ceiling LVOS $(count runs/jf/t3_val/cases)/140   MOSE $(count runs/jf/t3_mose/cases)/300   fp32 LVOS $(count runs/jf/fp32_val/cases)/20   MOSE $(count runs/jf/fp32_mose/cases)/60"
echo "[3] baselines and Base+ anchors LVOS $(count runs/jf/t4_val/cases)/140   MOSE $(count runs/jf/t4_mose/cases)/300   $( [ -f runs/reports/ALL_DONE3 ] && echo DONE )"
echo "[4] accuracy: gain and refresh MOSE $(count runs/jf/t5_mose/cases)/300   LVOS $(count runs/jf/t5_val/cases)/140   error parts MOSE $(count runs/jf/t6_mose/cases)/300   LVOS $(count runs/jf/t6_val/cases)/140   $( [ -f runs/reports/ALL_DONE4 ] && echo DONE )"
echo "load limit: $(cat runs/control/max_per_gpu 2>/dev/null) process(es) per GPU, duty $(cat runs/control/duty 2>/dev/null)%, governor $(pgrep -f 'code/governo[r].py' | wc -l), paused $(ps -eo stat,args | grep -E '^T.*code/(handoff_eval|more_eval|jf_eval|unseen_eval|unseen_more|train_spatial)\.py' | wc -l)"
echo '--- recent supervisor events'
tail -4 logs/pipeline/supervisor.log 2>/dev/null
tail -6 logs/pipeline2/supervisor.log 2>/dev/null
echo '--- gpu'
nvidia-smi --query-gpu=index,utilization.gpu,power.draw,temperature.gpu,fan.speed,memory.used --format=csv,noheader
