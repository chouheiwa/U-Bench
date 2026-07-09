#!/usr/bin/env bash
# 巡检 P1.4 标定组 9 个训练: 每 INTERVAL 秒汇报 完成数 / 各 job epoch / best val_iou / 崩溃信号。
# 全部到 250ep 后输出 DONE 并退出(结束 Monitor watch)。
# 用法: bash tools/watch_calib.sh [间隔秒=900]
cd /home/chouheiwa/python/U-Bench
INTERVAL="${1:-900}"
TOTAL=9
while true; do
  ts=$(date '+%H:%M:%S'); done=0; run=""; fin=""; err=""
  for d in output/USEANet/busi/calib_* output/USEANet/bus/calib_* output/USEANet/BUSBRA/calib_*; do
    [ -d "$d" ] || continue
    name=$(basename "$d"); log="$d/training.log"
    [ -f "$log" ] || { run="$run ${name}:init"; continue; }
    ep=$(grep -oE "epoch \[[0-9]+/250\]" "$log" | tail -1 | grep -oE "^epoch \[[0-9]+" | grep -oE "[0-9]+")
    best=$(grep -oE "val_iou [0-9.]+" "$log" | awk '{print $2}' | sort -rn | head -1)
    if grep -q "epoch \[249/250\]" "$log"; then
      done=$((done+1)); fin="$fin ${name#calib_}=${best}"
    else
      run="$run ${name#calib_}:e${ep:-0}/iou${best:-NA}"
    fi
    grep -qE "Traceback|CUDA out of memory|RuntimeError|Killed" "$log" 2>/dev/null && err="$err ${name#calib_}"
  done
  echo "[$ts] calib ${done}/${TOTAL} done | RUN:${run:- none} | FIN:${fin:- none} | ERR:${err:-none}"
  [ "$done" -ge "$TOTAL" ] && { echo "[$ts] === ALL ${TOTAL} CALIB RUNS DONE, best val_iou:${fin} ==="; break; }
  sleep "$INTERVAL"
done
