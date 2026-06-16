#!/usr/bin/env bash
# C3a baseline 对标 — seed41 满 250ep,同 BUSI split,对齐 USEANet 配方(强增广)。
# 用法: bash tools/run_baselines_s41.sh <GPU_INDEX> <queue_name>
# recipe 优化器门控在 model=='USEANet',baseline 自动走标准 SGD poly-LR。
set -u
cd /home/chouheiwa/python/U-Bench
mkdir -p logs/baseline

GPU="$1"; QUEUE="$2"
COMMON="--gpu 0 --base_dir hf_data/data/busi --dataset_name busi \
  --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed 41 --exp_name baseline_s41"

run() {
  local name="$1" mid="$2" sz="$3"
  local log="logs/baseline/${name}_s41.log"
  echo "[$(date '+%H:%M:%S')] GPU${GPU} START ${name} (id=${mid}, sz=${sz})" | tee -a "logs/baseline/queue_gpu${GPU}.log"
  CUDA_VISIBLE_DEVICES="${GPU}" USEANET_STRONG_AUG=1 \
    conda run -n ubench1 python main.py --model "${name}" --model_id "${mid}" --img_size "${sz}" \
    ${COMMON} > "${log}" 2>&1
  local rc=$?
  echo "[$(date '+%H:%M:%S')] GPU${GPU} DONE  ${name} rc=${rc}" | tee -a "logs/baseline/queue_gpu${GPU}.log"
}

case "$QUEUE" in
  gpu0)
    run U_Net 1 256
    run ResNet34UnetPlus 12 256
    run H2Former 108 256
    ;;
  gpu1)
    run AttU_Net 3 256
    run SwinUnet 13 224
    run VMUNet 95 256
    ;;
esac
echo "[$(date '+%H:%M:%S')] queue ${QUEUE} ALL DONE" | tee -a "logs/baseline/queue_gpu${GPU}.log"
