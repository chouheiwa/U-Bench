#!/usr/bin/env bash
# P1 深化 · 源依赖补验: 逐物理量消融从 busi + BUSBRA 源各跑一遍(bus 源已有)。
# 18 job = 2源(busi/BUSBRA) × 3物理量(nak/att/snr) × 3seed。
# 目的: 坐实"源退化程度 -> 衰减(att)跨域增益"的缩放关系(bus退化源att显著涨,
#   busi干净源疑似无效/有害)。busi->GPU0, BUSBRA->GPU1, 各卡 2 并发滚动。
# 完成后跑 run_phys_cross_matrix 的 busi/BUSBRA 变体做跨域评测 + wilcoxon_cross。
# 用法: bash tools/run_phys_matrix2.sh
set -u
cd /home/chouheiwa/python/U-Bench
mkdir -p logs/physabl

run_group() {  # $1=gpu $2=dataset, rest="phys:seed"...
  local gpu=$1 ds=$2; shift 2
  local base="./data/${ds}"
  local i=0
  for j in "$@"; do
    local phys=${j%:*} sd=${j#*:}
    echo "[$(date '+%H:%M:%S')] GPU${gpu} launch ${ds}/${phys}_s${sd}" | tee -a logs/physabl/queue2.log
    bash tools/run_phys_ablation.sh "$gpu" "$phys" "$ds" "$base" "$sd" &
    i=$((i+1))
    [ $((i % 2)) -eq 0 ] && wait
  done
  wait
}

JOBS="nakagami_m:41 nakagami_m:42 nakagami_m:43 attenuation:41 attenuation:42 attenuation:43 snr:41 snr:42 snr:43"
run_group 0 busi   $JOBS &
run_group 1 BUSBRA $JOBS &
wait
echo "=== PHYS MATRIX2 (busi+BUSBRA) DONE ($(date '+%H:%M:%S')) ===" | tee -a logs/physabl/queue2.log
