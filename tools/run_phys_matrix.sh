#!/usr/bin/env bash
# P1 深化 · 逐物理量消融并行调度: 9 job = 3 物理量(nak/att/snr) × 3 seed,均在 bus 源。
# bus 选作代表源:它跨域增益最大(bus->BrEaST +0.0296),归因信号最强。
# 各 job 独立,分 2 卡每卡 2 并发;单训练 ~1.4GB/20%,预计 ~3.5h 全清。
# 完成后接 run_calib_cross_matrix 风格的跨域评测,比 3 单量子集的跨域增益。
# 用法: bash tools/run_phys_matrix.sh
set -u
cd /home/chouheiwa/python/U-Bench
mkdir -p logs/physabl

BASE=./data/bus

# 组内 2 并发滚动。$1=gpu, rest="phys:seed"...
run_group() {
  local gpu=$1; shift
  local i=0
  for j in "$@"; do
    local phys=${j%:*} sd=${j#*:}
    echo "[$(date '+%H:%M:%S')] GPU${gpu} launch ${phys}_s${sd}" | tee -a logs/physabl/queue.log
    bash tools/run_phys_ablation.sh "$gpu" "$phys" bus "$BASE" "$sd" &
    i=$((i+1))
    [ $((i % 2)) -eq 0 ] && wait
  done
  wait
}

run_group 0 nakagami_m:41 nakagami_m:42 attenuation:41 attenuation:42 &
run_group 1 nakagami_m:43 attenuation:43 snr:41 snr:42 snr:43 &
wait
echo "=== PHYS ABLATION MATRIX DONE ($(date '+%H:%M:%S')) ===" | tee -a logs/physabl/queue.log
