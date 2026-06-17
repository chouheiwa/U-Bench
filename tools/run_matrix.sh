#!/usr/bin/env bash
# 全矩阵 baseline 自驱动器(work-stealing)。用法: bash tools/run_matrix.sh <GPU>
# 两卡各跑一个;共享任务表,mkdir 原子锁抢任务,CSV 查重跳过已完成(复用历史结果),
# 失败留 FAILED 标记不死循环。幂等、可重启。USEANet 不在表内(已全跑完且用自有配方)。
set -u
cd /home/chouheiwa/python/U-Bench
GPU="$1"
mkdir -p logs/matrix locks/matrix

# 方法: name|model_id|img_size  (SwinUnet 仅 224)
METHODS=(
  "U_Net|1|256" "AttU_Net|3|256" "ResNet34UnetPlus|12|256" "SwinUnet|13|224"
  "H2Former|108|256" "VMUNet|95|256" "TransUnet|10|256" "CMU_Net|4|256"
  "CMUNeXt|5|256" "MSLAU_Net|81|256"
)
# 数据集: dsname|base_dir
DATASETS=( "busi|hf_data/data/busi" "bus|hf_data/data/bus" "BUSBRA|hf_data/data/BUSBRA" "tuscui|hf_data/data/tuscui" )
SEEDS=( 41 42 43 )

# 已完成? CSV(result_<DS>_train.csv)有 model==m && seed==s && best_iou(field23)>0；或 FAILED 标记
is_done() {
  local m="$1" ds="$2" s="$3" lock="$4"
  [ -e "${lock}/FAILED" ] && return 0
  local csv="result/result_${ds}_train.csv"
  [ -f "$csv" ] || return 1
  awk -F, -v m="$m" -v s="$s" '$1==m && $10==s && ($23+0)>0 {f=1} END{exit !f}' "$csv"
}

while true; do
  remaining=0; ran=0
  for mt in "${METHODS[@]}"; do
    IFS='|' read -r name mid isz <<< "$mt"
    for d in "${DATASETS[@]}"; do
      IFS='|' read -r ds base <<< "$d"
      for s in "${SEEDS[@]}"; do
        lock="locks/matrix/${name}_${ds}_s${s}"
        is_done "$name" "$ds" "$s" "$lock" && continue
        remaining=1
        mkdir "$lock" 2>/dev/null || continue          # 被另一卡抢走/在跑
        if is_done "$name" "$ds" "$s" "$lock"; then rmdir "$lock" 2>/dev/null; continue; fi
        ran=1
        exp="mtx_s${s}"
        log="logs/matrix/${name}_${ds}_s${s}.log"
        echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} START ${name} ${ds} s${s}" | tee -a logs/matrix/driver_gpu${GPU}.log
        env CUDA_VISIBLE_DEVICES="${GPU}" USEANET_STRONG_AUG=1 \
          conda run -n ubench1 python -u main.py --model "${name}" --model_id "${mid}" --img_size "${isz}" \
          --gpu 0 --base_dir "${base}" --dataset_name "${ds}" \
          --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed "${s}" --exp_name "${exp}" > "${log}" 2>&1
        rc=$?
        if is_done "$name" "$ds" "$s" "$lock"; then
          rmdir "$lock" 2>/dev/null
          echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} DONE  ${name} ${ds} s${s} rc=${rc}" | tee -a logs/matrix/driver_gpu${GPU}.log
        else
          touch "${lock}/FAILED"   # 不死循环;留标记待人工查
          echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} FAIL  ${name} ${ds} s${s} rc=${rc} (see ${log})" | tee -a logs/matrix/driver_gpu${GPU}.log
        fi
      done
    done
  done
  [ "$remaining" -eq 0 ] && { echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} ALL DONE" | tee -a logs/matrix/driver_gpu${GPU}.log; break; }
  [ "$ran" -eq 0 ] && sleep 120   # 剩余都在另一卡上跑,歇会再扫
done
