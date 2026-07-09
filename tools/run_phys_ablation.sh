#!/usr/bin/env bash
# P1 深化: 逐物理量消融 = C2a 配方 + USEANET_CALIB_PROXY=1 + 单个物理量。
# 隔离 Nakagami / 衰减 / SNR 各自对(跨域)增益的贡献。
# 用法: bash tools/run_phys_ablation.sh <GPU> <phys> <dataset> <base_dir> <seed> [ep]
#   phys ∈ nakagami_m | attenuation | snr   (也可传两量逗号分隔)
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/physabl
GPU="$1"; PHYS="$2"; DS="$3"; BASE="$4"; SEED="$5"; EP="${6:-250}"
# 短标签 nak/att/snr 进 exp_name
case "$PHYS" in
  nakagami_m) TAG=nak ;; attenuation) TAG=att ;; snr) TAG=snr ;;
  *) TAG=$(echo "$PHYS" | tr -d ',_' | cut -c1-6) ;;
esac
EXP="calib${TAG}_${DS}_s${SEED}"
LOG="logs/physabl/${EXP}.log"
echo "[$(date '+%H:%M:%S')] GPU${GPU} START ${EXP} (phys=${PHYS}, ep=${EP})" | tee -a logs/physabl/queue.log
env CUDA_VISIBLE_DEVICES="${GPU}" USEANET_CALIB_PROXY=1 USEANET_CALIB_PHYS="${PHYS}" \
  USEANET_DISC_LR=1 USEANET_BACKBONE_LR_MULT=0.2 USEANET_STRONG_AUG=1 USEANET_LOSS_REGION=iou \
  conda run -n ubench1 python -u main.py --gpu 0 --model USEANet --model_id 115 \
  --base_dir "${BASE}" --dataset_name "${DS}" --do_deeps 1 \
  --pretrained_model_path ./pretrained \
  --batch_size 8 --max_epochs "${EP}" --base_lr 0.01 --seed "${SEED}" --exp_name "${EXP}" > "${LOG}" 2>&1
echo "[$(date '+%H:%M:%S')] GPU${GPU} DONE  ${EXP} rc=$?" | tee -a logs/physabl/queue.log
