#!/usr/bin/env bash
# P0.3 naive 跨域矩阵: 4 源 × 4 目标 × 3 seed, USEANet MoE-on 主配方 checkpoint。
# 对角线(源==目标)当 in-domain 复现 sanity check。结果落 result/result_cross_dataset.csv。
# 用法: bash tools/run_cross_dataset_matrix.sh [GPU]
set -u
cd "$(dirname "$0")/.."
GPU="${1:-0}"
DATASETS=(busi bus BUSBRA tuscui)

# 每个源 × seed 的 exp_name(已核实 checkpoint_best.pth 存在)
exp_name() {  # $1=source $2=seed
  case "$1_$2" in
    busi_41) echo disc_mult02_busi ;;
    busi_42) echo disc_mult02_s42 ;;
    busi_43) echo disc_mult02_s43 ;;
    *) echo "c2a_$1_s$2" ;;
  esac
}

for src in "${DATASETS[@]}"; do
  for seed in 41 42 43; do
    exp=$(exp_name "$src" "$seed")
    edir="./output/USEANet/$src/$exp"
    if [ ! -f "$edir/checkpoint_best.pth" ]; then
      echo "!! skip $src s$seed: no ckpt at $edir"; continue
    fi
    for tgt in "${DATASETS[@]}"; do
      echo "== $src -> $tgt (s$seed, $exp) =="
      CUDA_VISIBLE_DEVICES="$GPU" conda run -n ubench1 python tools/cross_dataset_eval.py \
        --source "$src" --base_dir "./data/$src" \
        --target "$tgt" --target_base_dir "./data/$tgt" \
        --seed "$seed" --exp_name "$exp" --exp_save_dir "$edir" 2>&1 \
        | grep -E "^\[xds\]|Error|Traceback|CUDA" | tail -3
    done
  done
done
echo "=== MATRIX DONE ==="
