#!/usr/bin/env bash
# 补算 8 个方法的 busi-s41 HD95 —— 它们用的是 baseline_s41 exp(非 mtx_s41)
set -u
GPU="${1:-0}"
# name|model_id|img_size
CELLS=(
  "U_Net|1|256" "AttU_Net|3|256" "ResNet34UnetPlus|12|256" "SwinUnet|13|224"
  "H2Former|108|256" "VMUNet|95|256" "TransUnet|10|256" "CMUNeXt|5|256"
)
mkdir -p logs/hd95
for c in "${CELLS[@]}"; do
  IFS='|' read -r name mid isz <<< "$c"
  exp="baseline_s41"
  expdir="output/${name}/${ds:-busi}/${exp}"
  expdir="output/${name}/busi/${exp}"
  ck="${expdir}/checkpoint_best.pth"
  if [ ! -f "$ck" ]; then echo "[SKIP] $name 无 $ck"; continue; fi
  log="logs/hd95/${name}_busi_s41_${exp}.log"
  echo "[RUN] $name busi s41 ($exp) ..."
  CUDA_VISIBLE_DEVICES="${GPU}" conda run -n ubench1 python tools/offline_hd95.py \
    --model "$name" --model_id "$mid" --img_size "$isz" \
    --base_dir "hf_data/data/busi" --dataset_name "busi" --seed 41 \
    --exp_save_dir "$expdir" --exp_name "$exp" --batch_size 1 > "$log" 2>&1
  rc=$?
  echo "[DONE rc=$rc] $name (log: $log)"
done
echo "HD95 busi_s41 FIX ALL DONE"
