#!/usr/bin/env bash
# P1.4 标定组训练并行调度: 9 个 job (busi/bus/BUSBRA 各 3 seed) 分 2 卡,每卡 2 并发。
# 各 job 独立无依赖;单训练仅 1.4GB/20%,双卡各2并发 ~3.5h 全清(vs 单卡串行 ~10h)。
# 手工基线组复用已有 c2a_*/disc_mult02_* checkpoint,不在此脚本。
# 用法: bash tools/run_calib_matrix.sh
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/calib

base_of() { echo "./data/$1"; }

# 每张卡一组,组内 2 并发滚动(满 2 个等一批完再起下一批)。
run_group() {  # $1=gpu, rest="ds:seed"...
  local gpu=$1; shift
  local i=0
  for j in "$@"; do
    local ds=${j%:*} sd=${j#*:}
    echo "[$(date '+%H:%M:%S')] GPU${gpu} launch calib_${ds}_s${sd}"
    bash tools/run_calib_job.sh "$gpu" "$ds" "$(base_of "$ds")" "$sd" &
    i=$((i+1))
    [ $((i % 2)) -eq 0 ] && wait   # 满 2 并发,等这批完再起下一批
  done
  wait
}

run_group 0 busi:42 bus:41 bus:43 BUSBRA:42 &
run_group 1 busi:41 busi:43 bus:42 BUSBRA:41 BUSBRA:43 &
wait
echo "=== CALIB MATRIX DONE ($(date '+%H:%M:%S')) ==="
