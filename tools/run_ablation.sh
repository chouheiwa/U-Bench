#!/usr/bin/env bash
# USEANet PhysicsMoE 消融驱动(work-stealing,双卡各跑一个实例)。
# 基线配方 = C2a/disc_mult02 满配方;每个消融只改一个旋钮。
# full 锚点(6 专家 / top-2 / 路由监督 on)= 已有 disc_mult02_s{41,42,43},不重跑。
# MoE on/off 这条已由 nomoe_mult02 vs disc_mult02 覆盖,不重跑。
#
# 用法(双卡并行):
#   nohup bash tools/run_ablation.sh 0 >/dev/null 2>&1 &
#   nohup bash tools/run_ablation.sh 1 >/dev/null 2>&1 &
set -u
cd /home/chouheiwa/python/U-Bench
GPU="${1:?用法: run_ablation.sh <GPU>}"
LOCKS="locks/ablation"; mkdir -p "$LOCKS" logs/ablation
PRETRAIN=/home/chouheiwa/experiment/pretrain_model
BASE_RECIPE="USEANET_DISC_LR=1 USEANET_BACKBONE_LR_MULT=0.2 USEANET_STRONG_AUG=1 USEANET_LOSS_REGION=iou"
DRIVERLOG="logs/ablation/driver_gpu${GPU}.log"

# 作业表: exp前缀|额外env|描述
JOBS=(
  "abl_ne2|USEANET_NUM_EXPERTS=2|2 physics experts"
  "abl_ne4|USEANET_NUM_EXPERTS=4|4 physics experts"
  "abl_top1|USEANET_TOPK=1|top-1 routing"
  "abl_nosup|USEANET_ROUTE_WEIGHT=0|router-supervision off"
)
SEEDS=(41 42 43)

log(){ echo "[$(date '+%m-%d %H:%M:%S')] GPU${GPU} $*" | tee -a "$DRIVERLOG"; }

log "START driver (base=hf_data/data/busi, 250ep)"
for job in "${JOBS[@]}"; do
  IFS='|' read -r prefix extra desc <<< "$job"
  for s in "${SEEDS[@]}"; do
    exp="${prefix}_s${s}"
    out="output/USEANet/busi/${exp}"
    [ -f "$LOCKS/${exp}.done" ] && { log "SKIP $exp (done)"; continue; }
    # 原子认领(mkdir 失败=另一卡已认领或正在跑)
    mkdir "$LOCKS/${exp}.claim" 2>/dev/null || continue
    LOG="logs/ablation/${exp}.log"
    log "RUN $exp ($desc | $extra)"
    env CUDA_VISIBLE_DEVICES="$GPU" $BASE_RECIPE $extra \
      conda run -n ubench1 python -u main.py --gpu 0 --model USEANet --model_id 115 \
      --base_dir hf_data/data/busi --dataset_name busi --do_deeps 1 \
      --pretrained_model_path "$PRETRAIN" \
      --batch_size 8 --max_epochs 250 --base_lr 0.01 --seed "$s" --exp_name "$exp" > "$LOG" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -f "$out/checkpoint_best.pth" ]; then
      touch "$LOCKS/${exp}.done"; log "DONE $exp rc=$rc"
    else
      touch "$LOCKS/${exp}.FAILED"; log "FAILED $exp rc=$rc (see $LOG)"
    fi
  done
done
log "ALL DONE"
