#!/usr/bin/env bash
# Cross-architecture reliability-gate matrix (paper rebuttal: "does the gate work
# on OTHER backbones?"). For each standard backbone, run naive zero-shot cross-
# domain transfer over the same source/target grid used for the USEANet gate, and
# dump per-case IoU + label-free signals (ent/margin/band/fgfrac) so the
# selective-prediction analysis can be reproduced per architecture.
#
# Mechanism 1 (selective prediction) only — inference, no training, no adaptation.
# Output: result/percase_gate_multiarch.csv  (modelname column groups by arch)
# Usage: bash tools/run_gate_multiarch.sh [GPU]
set -u
cd "$(dirname "$0")/.."
GPU="${1:-0}"

MODELS=(U_Net TransUnet SwinUnet)     # CNN / hybrid / pure-Transformer — max architectural spread
SOURCES=(busi bus BUSBRA)             # same sources as the USEANet gate (tools/failure_gate.py)
TARGETS=(busi bus BUSBRA BrEaST)      # BrEaST is target-only (no trained model)
PERCASE=./result/percase_gate_multiarch.csv
LOG=./result/_gate_multiarch.log

exp_name() {  # $1=source $2=seed  (standard-backbone checkpoint naming)
  case "$1_$2" in
    busi_41) echo baseline_s41 ;;
    busi_42) echo mtx_s42 ;;
    busi_43) echo mtx_s43 ;;
    *) echo "mtx_s$2" ;;              # bus/BUSBRA seed{41,42,43} -> mtx_s{41,42,43}
  esac
}

rm -f "$PERCASE"                       # idempotent: rebuild the whole per-case dump
: > "$LOG"
n_ok=0; n_skip=0; n_fail=0
for model in "${MODELS[@]}"; do
  for src in "${SOURCES[@]}"; do
    for seed in 41 42 43; do
      exp=$(exp_name "$src" "$seed")
      edir="./output/$model/$src/$exp"
      if [ ! -f "$edir/checkpoint_best.pth" ]; then
        echo "!! skip $model $src s$seed: no ckpt at $edir" | tee -a "$LOG"; n_skip=$((n_skip+1)); continue
      fi
      for tgt in "${TARGETS[@]}"; do
        out=$(CUDA_VISIBLE_DEVICES="$GPU" conda run -n ubench1 python tools/cross_dataset_eval.py \
          --model "$model" --pretrained_model_path "" \
          --source "$src" --base_dir "./data/$src" \
          --target "$tgt" --target_base_dir "./data/$tgt" \
          --seed "$seed" --exp_name "$exp" --exp_save_dir "$edir" \
          --dump_cases --percase_csv "$PERCASE" 2>&1)
        line=$(echo "$out" | grep -E "^\[xds\]" | tail -1)
        if [ -n "$line" ]; then echo "$line" | tee -a "$LOG"; n_ok=$((n_ok+1))
        else echo "!! FAIL $model $src->$tgt s$seed" | tee -a "$LOG"
             echo "$out" | grep -E "Error|Traceback|unexpected|missing|shape" | tail -2 | tee -a "$LOG"; n_fail=$((n_fail+1)); fi
      done
    done
  done
done
echo "=== GATE MULTIARCH DONE: ok=$n_ok skip=$n_skip fail=$n_fail -> $PERCASE ===" | tee -a "$LOG"
