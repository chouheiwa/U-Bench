#!/usr/bin/env bash
# nnU-Net Phase B smoke: bus seed41 5-epoch, 验证全链路
set -u
cd "$(dirname "$0")/.."
GPU="${1:-0}"
REPO="$(pwd)"
export nnUNet_raw="${REPO}/nnunet/raw"
export nnUNet_preprocessed="${REPO}/nnunet/preprocessed"
export nnUNet_results="${REPO}/nnunet/results"
export nnUNet_compile=f   # 关 torch.compile:cu118+torch2.7 下 compile 段错误杀进程(epoch0 秒退无 checkpoint)
mkdir -p logs/nnunet "$nnUNet_preprocessed" "$nnUNet_results"

ds="bus"; did="502"; s="41"
name=$(printf 'Dataset%03d_%s' "$did" "$ds")
tr_name="nnUNetTrainer_s${s}_5e"

echo "=== [1] convert ==="
[ -f "${nnUNet_raw}/${name}/.convert_done" ] || \
  conda run -n nnunet python tools/nnunet/convert_dataset.py --dataset "$ds"
echo "=== [2] plan_and_preprocess ==="
conda run -n nnunet nnUNetv2_plan_and_preprocess -d "$did" -c 2d --verify_dataset_integrity
echo "=== [3] inject splits_final (U-Bench fold0) ==="
cp "${nnUNet_raw}/${name}/ubench_split.json" \
   "${nnUNet_preprocessed}/${name}/splits_final.json"
echo "=== [4] train 5e ==="
pred="${nnUNet_results}/${name}/pred_s${s}_smoke"
rm -rf "$pred"; mkdir -p "$pred"
CUDA_VISIBLE_DEVICES="${GPU}" conda run -n nnunet nnUNetv2_train "$did" 2d 0 -tr "$tr_name" --npz
echo "=== [5] predict ==="
CUDA_VISIBLE_DEVICES="${GPU}" conda run -n nnunet nnUNetv2_predict \
    -i "${nnUNet_raw}/${name}/imagesVal" -o "$pred" \
    -d "$did" -c 2d -f 0 -tr "$tr_name" -chk checkpoint_best.pth
echo "=== [6] eval ==="
conda run -n nnunet python tools/nnunet/eval_nnunet.py --dataset "$ds" --seed "$s" --pred-dir "$pred"
echo "=== SMOKE DONE ==="
