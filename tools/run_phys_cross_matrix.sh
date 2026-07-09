#!/usr/bin/env bash
# P1 深化 · 逐物理量跨域评测: 4 变体 × 4 目标 × 3 seed = 48 次,源恒为 bus。
#   变体: nak/att/snr(单量子集) + full(全3量,已有 calib_bus_s*)。
# 关键: eval 时 build_model 按 env 建 estimator,单量子集的 to_proxy 通道数=1,
#   故必须同时设 USEANET_CALIB_PROXY=1 + USEANET_CALIB_PHYS 匹配训练子集,
#   否则 load_state_dict 形状不匹配。full 变体 USEANET_CALIB_PHYS 置空=全3量。
# 开 --dump_cases: 逐 case IoU 落 result/percase_cross_dataset.csv 供 Wilcoxon 配对检验。
# 非破坏(只读 checkpoint_best.pth)。须等对应训练跑完。
# 用法: bash tools/run_phys_cross_matrix.sh [GPU] [SRC]   (SRC 默认 bus)
set -u
cd "$(dirname "$0")/.."
GPU="${1:-0}"
SRC="${2:-bus}"
TGTS=(busi bus BUSBRA BrEaST)

# 变体名 -> "exp前缀:CALIB_PHYS值"  (full 的 phys 值为空串)
declare -A VAR
VAR[nak]="calibnak:nakagami_m"
VAR[att]="calibatt:attenuation"
VAR[snr]="calibsnr:snr"
VAR[full]="calib:"

for v in nak att snr full; do
  pre=${VAR[$v]%:*}; phys=${VAR[$v]#*:}
  for seed in 41 42 43; do
    exp="${pre}_${SRC}_s${seed}"
    edir="./output/USEANet/${SRC}/${exp}"
    if [ ! -f "$edir/checkpoint_best.pth" ]; then
      echo "!! skip $exp: no checkpoint_best.pth"; continue
    fi
    for tgt in "${TGTS[@]}"; do
      echo "== [$v] $SRC -> $tgt (s$seed, phys='${phys:-all3}') =="
      CUDA_VISIBLE_DEVICES="$GPU" USEANET_CALIB_PROXY=1 USEANET_CALIB_PHYS="$phys" \
        conda run -n ubench1 python tools/cross_dataset_eval.py \
        --source "$SRC" --base_dir "./data/$SRC" \
        --target "$tgt" --target_base_dir "./data/$tgt" \
        --seed "$seed" --exp_name "$exp" --exp_save_dir "$edir" --dump_cases 2>&1 \
        | grep -E "^\[xds\]|Error|Traceback|CUDA|size mismatch" | tail -3
    done
  done
done
echo "=== PHYS CROSS MATRIX DONE ==="
