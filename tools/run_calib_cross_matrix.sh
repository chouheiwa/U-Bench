#!/usr/bin/env bash
# P1.4 标定组跨域评测: 3源(busi/bus/BUSBRA)×4目标(busi/bus/BUSBRA/BrEaST)×3seed = 36 次。
# 用 calib_<src>_s<seed> 的 checkpoint_best.pth。非破坏性(只读 ckpt)。
# 结果落 result/result_cross_dataset.csv,exp_name 前缀 calib_ 与手工基线(disc_mult02_/c2a_)区分。
# 手工基线同子矩阵已在 csv 中,聚合时按 exp_name 前缀对比。
# 用法: bash tools/run_calib_cross_matrix.sh [GPU]   (须等 9 个标定训练全部跑完)
set -u
cd /home/chouheiwa/python/U-Bench
GPU="${1:-0}"
SRCS=(busi bus BUSBRA)
TGTS=(busi bus BUSBRA BrEaST)

for src in "${SRCS[@]}"; do
  for seed in 41 42 43; do
    exp="calib_${src}_s${seed}"
    edir="./output/USEANet/${src}/${exp}"
    if [ ! -f "$edir/checkpoint_best.pth" ]; then
      echo "!! skip $exp: no checkpoint_best.pth"; continue
    fi
    for tgt in "${TGTS[@]}"; do
      echo "== $src -> $tgt (s$seed, calib) =="
      CUDA_VISIBLE_DEVICES="$GPU" conda run -n ubench1 python tools/cross_dataset_eval.py \
        --source "$src" --base_dir "./data/$src" \
        --target "$tgt" --target_base_dir "./data/$tgt" \
        --seed "$seed" --exp_name "$exp" --exp_save_dir "$edir" 2>&1 \
        | grep -E "^\[xds\]|Error|Traceback|CUDA" | tail -3
    done
  done
done
echo "=== CALIB CROSS MATRIX DONE ==="
