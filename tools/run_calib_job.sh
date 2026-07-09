#!/usr/bin/env bash
# P1.4 标定物理路由消融 = C2a 满配方 + USEANET_CALIB_PROXY=1
# (手工 degradation_proxies -> 学习式 PhysicsEstimator + Nakagami/衰减/SNR 弱监督)。
# 手工基线组直接复用已有 c2a_<ds>_s<seed> checkpoint,无需重训;本脚本只跑标定组。
# 用法: bash tools/run_calib_job.sh <GPU> <dataset_name> <base_dir> <seed> [max_epochs]
set -u
cd /home/chouheiwa/python/U-Bench
mkdir -p logs/calib
GPU="$1"; DS="$2"; BASE="$3"; SEED="$4"; EP="${5:-250}"
EXP="calib_${DS}_s${SEED}"
LOG="logs/calib/${EXP}.log"
echo "[$(date '+%H:%M:%S')] GPU${GPU} START ${EXP} (base=${BASE}, ep=${EP})" | tee -a logs/calib/queue.log
env CUDA_VISIBLE_DEVICES="${GPU}" USEANET_CALIB_PROXY=1 USEANET_DISC_LR=1 USEANET_BACKBONE_LR_MULT=0.2 \
  USEANET_STRONG_AUG=1 USEANET_LOSS_REGION=iou \
  conda run -n ubench1 python -u main.py --gpu 0 --model USEANet --model_id 115 \
  --base_dir "${BASE}" --dataset_name "${DS}" --do_deeps 1 \
  --pretrained_model_path /home/chouheiwa/experiment/pretrain_model \
  --batch_size 8 --max_epochs "${EP}" --base_lr 0.01 --seed "${SEED}" --exp_name "${EXP}" > "${LOG}" 2>&1
echo "[$(date '+%H:%M:%S')] GPU${GPU} DONE  ${EXP} rc=$?" | tee -a logs/calib/queue.log
