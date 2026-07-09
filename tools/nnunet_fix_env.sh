#!/usr/bin/env bash
# 把 nnunet 环境的 torch 换成 cu118 构建(对齐 ubench1,驱动 570 已验证可跑)+ 补 opencv
set -u
echo "=== [1] 重装 torch/torchvision cu118 ==="
conda run -n nnunet pip install --no-cache-dir \
  torch==2.7.0+cu118 torchvision==0.22.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
rc1=$?
echo "=== [2] 装 opencv-python-headless ==="
conda run -n nnunet pip install --no-cache-dir opencv-python-headless
rc2=$?
echo "=== [3] 验证 ==="
conda run -n nnunet python -c "import torch,cv2; print('torch',torch.__version__,'cuda_build',torch.version.cuda,'avail',torch.cuda.is_available()); print('cv2',cv2.__version__)"
echo "FIX_ENV rc1=$rc1 rc2=$rc2 DONE"
