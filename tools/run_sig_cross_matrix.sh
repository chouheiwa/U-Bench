#!/usr/bin/env bash
# P1 深化 · 12/12「标定 vs 手工」跨域显著性 re-dump:
#   2 组(calib / manual) × 3 源(busi/bus/BUSBRA) × 4 目标 × 3 seed = 72 次评测。
#   每 case IoU 落 result/percase_cross_dataset.csv,供 Wilcoxon 配对检验。
# 组与 env:
#   calib  : USEANET_CALIB_PROXY=1 (标定 PhysicsEstimator, 全3量)
#   manual : 不设 CALIB_PROXY (手工 degradation_proxies)
# 手工 exp_name 跨源不规则(busi seed41 = disc_mult02_busi,无 _s41 后缀):
#   busi:   s41=disc_mult02_busi  s42=disc_mult02_s42  s43=disc_mult02_s43
#   bus:    c2a_bus_s{seed}
#   BUSBRA: c2a_BUSBRA_s{seed}
# 非破坏(只读 checkpoint_best.pth)。须等 calib_BUSBRA_s43 补训完成。
# 用法: bash tools/run_sig_cross_matrix.sh [GPU]
set -u
cd /home/chouheiwa/python/U-Bench
GPU="${1:-0}"
SRCS=(busi bus BUSBRA)
TGTS=(busi bus BUSBRA BrEaST)

# 手工组 exp_name 解析(处理 busi 不规则命名)
manual_exp() {  # $1=src $2=seed
  case "$1" in
    busi)   [ "$2" = 41 ] && echo "disc_mult02_busi" || echo "disc_mult02_s$2" ;;
    bus)    echo "c2a_bus_s$2" ;;
    BUSBRA) echo "c2a_BUSBRA_s$2" ;;
  esac
}

eval_cell() {  # $1=group $2=src $3=seed $4=exp $5=calib_proxy(0/1)
  local grp=$1 src=$2 seed=$3 exp=$4 cp=$5
  local edir="./output/USEANet/${src}/${exp}"
  if [ ! -f "$edir/checkpoint_best.pth" ]; then
    echo "!! skip [$grp] $exp: no checkpoint_best.pth"; return
  fi
  for tgt in "${TGTS[@]}"; do
    echo "== [$grp] $src -> $tgt (s$seed, $exp) =="
    local envs="CUDA_VISIBLE_DEVICES=$GPU"
    if [ "$cp" = 1 ]; then envs="$envs USEANET_CALIB_PROXY=1 USEANET_CALIB_PHYS="; fi
    env $envs conda run -n ubench1 python tools/cross_dataset_eval.py \
      --source "$src" --base_dir "./data/$src" \
      --target "$tgt" --target_base_dir "./data/$tgt" \
      --seed "$seed" --exp_name "$exp" --exp_save_dir "$edir" --dump_cases 2>&1 \
      | grep -E "^\[xds\]|Error|Traceback|size mismatch|Unexpected|Missing key" | tail -3
  done
}

for src in "${SRCS[@]}"; do
  for seed in 41 42 43; do
    eval_cell calib  "$src" "$seed" "calib_${src}_s${seed}" 1
    eval_cell manual "$src" "$seed" "$(manual_exp "$src" "$seed")" 0
  done
done
echo "=== SIG CROSS MATRIX DONE ==="
