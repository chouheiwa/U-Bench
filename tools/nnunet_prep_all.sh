#!/usr/bin/env bash
# 串行预备全部 4 数据集(canonical nnU-Net:先一次性预处理,再并行训练),避免并发预处理竞争
set -u
cd "$(dirname "$0")/.."
REPO="$(pwd)"
export nnUNet_raw="${REPO}/nnunet/raw"
export nnUNet_preprocessed="${REPO}/nnunet/preprocessed"
export nnUNet_results="${REPO}/nnunet/results"
export nnUNet_compile=f
DATASETS=( "busi|501" "bus|502" "BUSBRA|503" "tuscui|504" )
for d in "${DATASETS[@]}"; do
  IFS='|' read -r ds did <<< "$d"
  name=$(printf 'Dataset%03d_%s' "$did" "$ds")
  sent="${nnUNet_preprocessed}/${name}/.ubench_ready"
  echo "===== PREP ${name} ====="
  if [ -f "$sent" ]; then echo "[$name] 已 READY 跳过"; continue; fi
  # convert(幂等)
  if [ ! -f "${nnUNet_raw}/${name}/.convert_done" ]; then
    echo "[$name] convert..."; conda run -n nnunet python tools/nnunet/convert_dataset.py --dataset "$ds" || { echo "[$name] CONVERT_FAIL"; continue; }
  else echo "[$name] convert 已完成"; fi
  # plan + preprocess(串行,verify)
  echo "[$name] plan_and_preprocess..."
  conda run -n nnunet nnUNetv2_plan_and_preprocess -d "$did" -c 2d --verify_dataset_integrity || { echo "[$name] PREP_FAIL"; continue; }
  # 注入 U-Bench split 为 fold0
  cp "${nnUNet_raw}/${name}/ubench_split.json" "${nnUNet_preprocessed}/${name}/splits_final.json" || { echo "[$name] SPLIT_FAIL"; continue; }
  # 校验 .b2nd 数量(粗略:应≈训练+验证 case 数)
  nb=$(ls "${nnUNet_preprocessed}/${name}/nnUNetPlans_2d/"*.b2nd 2>/dev/null | wc -l)
  echo "[$name] .b2nd 文件数=${nb}"
  touch "$sent"
  echo "[$name] READY (sentinel 已建)"
done
echo "PREP_ALL DONE"
