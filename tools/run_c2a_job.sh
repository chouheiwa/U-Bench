#!/usr/bin/env bash
# C2a 跨数据集泛化单跑(USEANet 最佳同构配方 = disc+mult02+strong_aug+iou)。
# 用法: bash tools/run_c2a_job.sh <GPU> <dataset_name> <base_dir> <seed>
set -u
cd /home/chouheiwa/python/U-Bench
mkdir -p logs/c2a
GPU="$1"; DS="$2"; BASE="$3"; SEED="$4"
EXP="c2a_${DS}_s${SEED}"
LOG="logs/c2a/${EXP}.log"
echo "[$(date '+%H:%M:%S')] GPU${GPU} START ${EXP} (base=${BASE})" | tee -a logs/c2a/queue.log
env CUDA_VISIBLE_DEVICES="${GPU}" USEANET_DISC_LR=1 USEANET_BACKBONE_LR_MULT=0.2 \
  USEANET_STRONG_AUG=1 USEANET_LOSS_REGION=iou \
  conda run -n ubench1 python -u main.py --gpu 0 --model USEANet --model_id 115 \
  --base_dir "${BASE}" --dataset_name "${DS}" --do_deeps 1 \
  --pretrained_model_path /home/chouheiwa/experiment/pretrain_model \
  --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed "${SEED}" --exp_name "${EXP}" > "${LOG}" 2>&1
echo "[$(date '+%H:%M:%S')] GPU${GPU} DONE  ${EXP} rc=$?" | tee -a logs/c2a/queue.log
