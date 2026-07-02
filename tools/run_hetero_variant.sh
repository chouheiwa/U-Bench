#!/usr/bin/env bash
# 异构专家"去过拟合"变体单跑。用法: bash tools/run_hetero_variant.sh <GPU> <exp_name> [额外 env...]
# 额外 env 以 KEY=VAL 形式传(如 USEANET_HETERO_CHANNEL=32 USEANET_HETERO_NO_SE=1)。
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs/hetero
GPU="$1"; EXP="$2"; shift 2
EXTRA_ENV="$*"
LOG="logs/hetero/${EXP}.log"
echo "[$(date '+%H:%M:%S')] GPU${GPU} START ${EXP}  extra=[${EXTRA_ENV}]" | tee -a logs/hetero/queue.log
# `env` is needed so the word-split ${EXTRA_ENV} is parsed as KEY=VAL assignments
# (a bare expansion after assignment prefixes is NOT re-recognised as assignments).
env CUDA_VISIBLE_DEVICES="${GPU}" USEANET_HETERO_EXPERTS=1 USEANET_DISC_LR=1 USEANET_BACKBONE_LR_MULT=0.2 \
  USEANET_STRONG_AUG=1 USEANET_LOSS_REGION=iou ${EXTRA_ENV} \
  conda run -n ubench1 python -u main.py --gpu 0 --model USEANet --model_id 115 \
  --base_dir hf_data/data/busi --dataset_name busi --do_deeps 1 \
  --pretrained_model_path ${PRETRAINED_MODEL_PATH:-./pretrained} \
  --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed 41 --exp_name "${EXP}" > "${LOG}" 2>&1
echo "[$(date '+%H:%M:%S')] GPU${GPU} DONE  ${EXP} rc=$?" | tee -a logs/hetero/queue.log
