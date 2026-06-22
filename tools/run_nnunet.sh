#!/usr/bin/env bash
# nnU-Net v2 baseline 自驱动器(work-stealing)。用法: bash tools/run_nnunet.sh <GPU>
# 任务表 = 4 数据集 × 3 seed = 12 格。原生管线 + U-Bench split(splits_final.json 注入)。
# GPU 护栏:等本卡空(<2500MiB)再开 → 自动排在 120 格矩阵之后,不与之争卡。
# 幂等可重启:结果写 result/result_nnunet.csv;失败留 FAILED 标记不死循环。
# 唯一偏离 nnU-Net 默认:250 epoch(可行性 + 与 U-Bench 训练预算对齐)。
set -u
cd /home/chouheiwa/python/U-Bench
GPU="$1"

REPO="$(pwd)"
export nnUNet_raw="${REPO}/nnunet/raw"
export nnUNet_preprocessed="${REPO}/nnunet/preprocessed"
export nnUNet_results="${REPO}/nnunet/results"
mkdir -p logs/nnunet locks/nnunet "$nnUNet_preprocessed" "$nnUNet_results"

# dsname|DatasetID
DATASETS=( "busi|501" "bus|502" "BUSBRA|503" "tuscui|504" )
SEEDS=( 41 42 43 )

NN() { conda run -n nnunet "$@"; }

# 已完成? result_nnunet.csv 有 dataset==ds && seed==s（best_iou>0）；或 FAILED 标记。
is_done() {
  local ds="$1" s="$2" lock="$3"
  [ -e "${lock}/FAILED" ] && return 0
  local csv="result/result_nnunet.csv"
  [ -f "$csv" ] || return 1
  awk -F, -v ds="$ds" -v s="$s" '
    $1=="nnUNet" && $2==ds && $3==s { if($4+0>0) f=1 }
    END{exit !f}' "$csv"
}

# 每数据集只做一次:转换 + plan_and_preprocess + 注入 splits_final.json(sentinel 守护)
ensure_dataset_ready() {
  local ds="$1" did="$2"
  local name; name=$(printf 'Dataset%03d_%s' "$did" "$ds")
  local sent="${nnUNet_preprocessed}/${name}/.ubench_ready"
  [ -f "$sent" ] && return 0
  # 转换(幂等)
  [ -f "${nnUNet_raw}/${name}/.convert_done" ] || \
    NN python tools/nnunet/convert_dataset.py --dataset "$ds" >> "logs/nnunet/prep_${ds}.log" 2>&1
  # 规划+预处理(2d 即可)
  NN nnUNetv2_plan_and_preprocess -d "$did" -c 2d --verify_dataset_integrity \
    >> "logs/nnunet/prep_${ds}.log" 2>&1 || return 1
  # 注入 U-Bench split 为 fold0
  cp "${nnUNet_raw}/${name}/ubench_split.json" \
     "${nnUNet_preprocessed}/${name}/splits_final.json" || return 1
  touch "$sent"
}

gpu_wait_free() {
  while true; do
    local used
    used=$(nvidia-smi -i "${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$used" ] && [ "$used" -lt 2500 ] && break
    sleep 60
  done
}

while true; do
  remaining=0; ran=0
  for d in "${DATASETS[@]}"; do
    IFS='|' read -r ds did <<< "$d"
    name=$(printf 'Dataset%03d_%s' "$did" "$ds")
    for s in "${SEEDS[@]}"; do
      lock="locks/nnunet/${ds}_s${s}"
      is_done "$ds" "$s" "$lock" && continue
      remaining=1
      mkdir "$lock" 2>/dev/null || continue
      if is_done "$ds" "$s" "$lock"; then rmdir "$lock" 2>/dev/null; continue; fi
      ran=1
      tr_name="nnUNetTrainer_s${s}_250e"
      log="logs/nnunet/${ds}_s${s}.log"
      gpu_wait_free
      echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} START ${ds} s${s}" | tee -a logs/nnunet/driver_gpu${GPU}.log
      {
        # 数据准备(每卡安全:sentinel + 内部各步幂等;preprocess 并发风险低,仅首格触发)
        ensure_dataset_ready "$ds" "$did" || { echo "PREP_FAILED"; }
        pred="${nnUNet_results}/${name}/pred_s${s}"
        rm -rf "$pred"; mkdir -p "$pred"
        CUDA_VISIBLE_DEVICES="${GPU}" conda run -n nnunet nnUNetv2_train "$did" 2d 0 -tr "$tr_name" --npz && \
        CUDA_VISIBLE_DEVICES="${GPU}" conda run -n nnunet nnUNetv2_predict \
            -i "${nnUNet_raw}/${name}/imagesVal" -o "$pred" \
            -d "$did" -c 2d -f 0 -tr "$tr_name" -chk checkpoint_best.pth && \
        conda run -n nnunet python tools/nnunet/eval_nnunet.py --dataset "$ds" --seed "$s" --pred-dir "$pred"
      } > "${log}" 2>&1
      rc=$?
      if is_done "$ds" "$s" "$lock"; then
        rmdir "$lock" 2>/dev/null
        echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} DONE  ${ds} s${s} rc=${rc}" | tee -a logs/nnunet/driver_gpu${GPU}.log
      else
        touch "${lock}/FAILED"
        echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} FAIL  ${ds} s${s} rc=${rc} (see ${log})" | tee -a logs/nnunet/driver_gpu${GPU}.log
      fi
    done
  done
  [ "$remaining" -eq 0 ] && { echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} ALL DONE" | tee -a logs/nnunet/driver_gpu${GPU}.log; break; }
  [ "$ran" -eq 0 ] && sleep 120
done
