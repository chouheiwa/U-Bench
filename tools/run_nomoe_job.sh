#!/usr/bin/env bash
# 实验 #7 跨集 MoE-off 消融单跑 = C2a 满配方 + USEANET_NO_MOE=1(PhysicsMoE→MultiBranchFeatureProcessor)。
# 用法: bash tools/run_nomoe_job.sh <GPU> <dataset_name> <base_dir> <seed>
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/nomoe
GPU="$1"; DS="$2"; BASE="$3"; SEED="$4"
EXP="nomoe_${DS}_s${SEED}"
LOG="logs/nomoe/${EXP}.log"
echo "[$(date '+%H:%M:%S')] GPU${GPU} START ${EXP} (base=${BASE})" | tee -a logs/nomoe/queue.log
env CUDA_VISIBLE_DEVICES="${GPU}" USEANET_NO_MOE=1 USEANET_DISC_LR=1 USEANET_BACKBONE_LR_MULT=0.2 \
  USEANET_STRONG_AUG=1 USEANET_LOSS_REGION=iou \
  conda run -n ubench1 python -u main.py --gpu 0 --model USEANet --model_id 115 \
  --base_dir "${BASE}" --dataset_name "${DS}" --do_deeps 1 \
  --pretrained_model_path ./pretrained \
  --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed "${SEED}" --exp_name "${EXP}" > "${LOG}" 2>&1
echo "[$(date '+%H:%M:%S')] GPU${GPU} DONE  ${EXP} rc=$?" | tee -a logs/nomoe/queue.log
