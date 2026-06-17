#!/usr/bin/env bash
set -u
cd /home/chouheiwa/python/U-Bench
GPU="$1"; NAME="$2"; MID="$3"
LOG="logs/baseline/${NAME}_s41.log"
echo "[$(date '+%H:%M:%S')] GPU${GPU} START ${NAME} (id=${MID})" | tee -a logs/baseline/extra_queue.log
env CUDA_VISIBLE_DEVICES="${GPU}" USEANET_STRONG_AUG=1 \
  conda run -n ubench1 python -u main.py --model "${NAME}" --model_id "${MID}" --img_size 256 \
  --gpu 0 --base_dir hf_data/data/busi --dataset_name busi \
  --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed 41 --exp_name baseline_s41 > "${LOG}" 2>&1
echo "[$(date '+%H:%M:%S')] GPU${GPU} DONE  ${NAME} rc=$?" | tee -a logs/baseline/extra_queue.log
