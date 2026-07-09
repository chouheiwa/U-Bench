"""跑 PUMA-Net(USEANet)在架构图所用的同一张超声图上的真实预测,
渲成 Lesion Mask 输出图(叠加 + 纯掩膜)。CPU 推理,不碰 GPU。
"""
import os, sys
from types import SimpleNamespace
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/chouheiwa/python/U-Bench"
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "paper/puma-net/assets")
os.makedirs(OUT, exist_ok=True)

IMG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "hf_data/data/busi/images/malignant (106).png")
EXPDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "output/USEANet/busi/moe_busi_strongaug_e250")
S = 256
IMEAN = np.array([0.485, 0.456, 0.406]); ISTD = np.array([0.229, 0.224, 0.225])


def main():
    args = SimpleNamespace(
        model="USEANet", model_id=115, img_size=S,
        base_dir="hf_data/data/busi", dataset_name="busi",
        batch_size=1, seed=41, input_channel=3, num_classes=1, do_deeps=False,
        pretrained_model_path=None, exp_save_dir=EXPDIR,
        train_file_dir="train.txt", val_file_dir="val.txt",
    )
    from models import build_model
    model = build_model(args, input_channel=3, num_classes=1).to("cpu")
    ckpt = torch.load(os.path.join(EXPDIR, "checkpoint_best.pth"), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    bgr = cv2.resize(cv2.imread(IMG, cv2.IMREAD_COLOR), (S, S))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(((rgb - IMEAN) / ISTD).transpose(2, 0, 1)).float().unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        prob = torch.sigmoid(out)[0, 0].cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8)
    print(f"[pred] lesion pixels = {int(mask.sum())} / {S*S}")

    # 纯二值掩膜(白病灶黑背景)
    cv2.imwrite(os.path.join(OUT, "pred_mask.png"), mask * 255)

    # GT 掩膜(同名,masks/0/ 下)
    gt_path = IMG.replace("/images/", "/masks/0/")
    gtm = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt = (cv2.resize(gtm, (S, S), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8) if gtm is not None else None

    # 叠加图:灰度底 + 红色半透明预测 + 红色预测轮廓 + 绿色 GT 轮廓
    base = np.stack([gray] * 3, -1).astype(np.float32)
    overlay = base.copy()
    overlay[mask == 1] = (0.45 * np.array([255, 60, 60]) + 0.55 * base[mask == 1])
    overlay = overlay.astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (255, 40, 40), 2)        # red = prediction
    if gt is not None:
        gcnts, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, gcnts, -1, (40, 220, 40), 2)   # green = ground truth
        print(f"[gt] gt pixels = {int(gt.sum())}")
    cv2.imwrite(os.path.join(OUT, "pred_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("PRED DONE ->", OUT)


if __name__ == "__main__":
    main()
