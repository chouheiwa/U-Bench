#!/usr/bin/env bash
# HD95 离线补算驱动(work-stealing)。用法: bash tools/run_hd95.sh <GPU>
# 对已训完的 best checkpoint 跑一次 val 推理算 HD95(无需重训)。排在 120 格矩阵之后:
# GPU 护栏等本卡空(<2500MiB)再开。幂等可重启,结果写 result/result_hd95.csv。
# 覆盖:10 基线的 mtx_ 格 + 可选 PUMA 清单(tools/hd95_puma_cells.txt)。
# 注:HD95 推理很快(单次 val 前向,分钟级/格),不是训练那种小时级。
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="$1"
mkdir -p logs/hd95 locks/hd95

# name|model_id|img_size  —— 与 run_matrix.sh 完全一致
METHODS=(
  "U_Net|1|256" "AttU_Net|3|256" "ResNet34UnetPlus|12|256" "SwinUnet|13|224"
  "H2Former|108|256" "VMUNet|95|256" "TransUnet|10|256" "CMU_Net|4|256"
  "CMUNeXt|5|256" "MSLAU_Net|81|256"
)
DATASETS=( "busi|hf_data/data/busi" "bus|hf_data/data/bus" "BUSBRA|hf_data/data/BUSBRA" "tuscui|hf_data/data/tuscui" )
SEEDS=( 41 42 43 )

is_done() {
  local model="$1" ds="$2" s="$3" exp="$4" lock="$5"
  [ -e "${lock}/FAILED" ] && return 0
  local csv="result/result_hd95.csv"
  [ -f "$csv" ] || return 1
  awk -F, -v m="$model" -v ds="$ds" -v s="$s" -v e="$exp" \
    '$1==m && $2==ds && $3==s && $4==e { if($5+0>=0 && $5!="") f=1 } END{exit !f}' "$csv"
}

gpu_wait_free() {
  while true; do
    local used
    used=$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$used" ] && [ "$used" -lt 2500 ] && break
    sleep 60
  done
}

# 一个单元: model_id|name|img_size|base_dir|dataset|seed|exp_save_dir|exp_name
run_cell() {
  IFS='|' read -r mid name isz base ds s expdir exp <<< "$1"
  local lock="locks/hd95/${name}_${ds}_s${s}_${exp}"
  is_done "$name" "$ds" "$s" "$exp" "$lock" && return 0
  [ -f "${expdir}/checkpoint_best.pth" ] || return 0   # 还没训完,跳过
  mkdir "$lock" 2>/dev/null || return 0
  if is_done "$name" "$ds" "$s" "$exp" "$lock"; then rmdir "$lock" 2>/dev/null; return 0; fi
  RAN=1
  local log="logs/hd95/${name}_${ds}_s${s}_${exp}.log"
  gpu_wait_free
  echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} HD95 START ${name} ${ds} s${s} ${exp}" | tee -a logs/hd95/driver_gpu${GPU}.log
  CUDA_VISIBLE_DEVICES="${GPU}" conda run -n ubench1 python tools/offline_hd95.py \
    --model "$name" --model_id "$mid" --img_size "$isz" \
    --base_dir "$base" --dataset_name "$ds" --seed "$s" \
    --exp_save_dir "$expdir" --exp_name "$exp" --batch_size 1 > "$log" 2>&1
  local rc=$?
  if is_done "$name" "$ds" "$s" "$exp" "$lock"; then
    rmdir "$lock" 2>/dev/null
    echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} HD95 DONE  ${name} ${ds} s${s} ${exp} rc=${rc}" | tee -a logs/hd95/driver_gpu${GPU}.log
  else
    touch "${lock}/FAILED"
    echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} HD95 FAIL  ${name} ${ds} s${s} ${exp} rc=${rc} (see ${log})" | tee -a logs/hd95/driver_gpu${GPU}.log
  fi
}

while true; do
  REMAIN=0; RAN=0
  # 10 基线 mtx_ 格
  for mt in "${METHODS[@]}"; do
    IFS='|' read -r name mid isz <<< "$mt"
    for d in "${DATASETS[@]}"; do
      IFS='|' read -r ds base <<< "$d"
      for s in "${SEEDS[@]}"; do
        exp="mtx_s${s}"; expdir="output/${name}/${ds}/${exp}"
        [ -f "${expdir}/checkpoint_best.pth" ] || continue
        lock="locks/hd95/${name}_${ds}_s${s}_${exp}"
        is_done "$name" "$ds" "$s" "$exp" "$lock" || REMAIN=1
        run_cell "${mid}|${name}|${isz}|${base}|${ds}|${s}|${expdir}|${exp}"
      done
    done
  done
  # 可选 PUMA 清单(每行: model_id|name|img_size|base_dir|dataset|seed|exp_save_dir|exp_name)
  if [ -f tools/hd95_puma_cells.txt ]; then
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      case "$line" in \#*) continue;; esac
      IFS='|' read -r mid name isz base ds s expdir exp <<< "$line"
      [ -f "${expdir}/checkpoint_best.pth" ] || continue
      lock="locks/hd95/${name}_${ds}_s${s}_${exp}"
      is_done "$name" "$ds" "$s" "$exp" "$lock" || REMAIN=1
      run_cell "$line"
    done < tools/hd95_puma_cells.txt
  fi
  [ "$REMAIN" -eq 0 ] && { echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} HD95 ALL DONE" | tee -a logs/hd95/driver_gpu${GPU}.log; break; }
  [ "$RAN" -eq 0 ] && sleep 120
done
