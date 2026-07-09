"""渲染架构图用的真实素材:真超声图 + PVT-B0 真特征图 + 真退化 proxy 热图。
CPU 推理,不碰 GPU。输出 paper/puma-net/assets/。
"""
import os, sys
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from models.Hybrid.USEANet.pvtv2 import pvt_v2_b0
from models.Hybrid.USEANet.moe.proxy import degradation_proxies
from models.Hybrid.USEANet.moe import EXPERT_NAMES

OUT = os.path.join(REPO, "paper/puma-net/assets")
os.makedirs(OUT, exist_ok=True)

IMG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "hf_data/data/busi/images/malignant (106).png")
CKPT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "output/USEANet/busi/moe_busi_strongaug_e250/checkpoint_best.pth")
S = 256
IMEAN = np.array([0.485, 0.456, 0.406]); ISTD = np.array([0.229, 0.224, 0.225])


def save_heat(arr, path, cmap):
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.ptp() + 1e-8)
    plt.imsave(path, arr, cmap=cmap, vmin=0, vmax=1)


def main():
    # --- 读图(灰度存档 + 3通道归一化喂网络)
    bgr = cv2.imread(IMG, cv2.IMREAD_COLOR)
    bgr = cv2.resize(bgr, (S, S), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(OUT, "input_us.png"), gray)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (rgb - IMEAN) / ISTD
    x = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0)  # [1,3,256,256]

    # --- 加载训练好的 PVT-B0 backbone(真实学到的特征)
    bb = pvt_v2_b0()
    sd = torch.load(CKPT, map_location="cpu")
    sd = sd.get("state_dict", sd)
    bsd = {k.split("backbone.", 1)[1]: v for k, v in sd.items() if "backbone." in k}
    missing = bb.load_state_dict(bsd, strict=False)
    print(f"[backbone] loaded {len(bsd)} keys; missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}")
    bb.eval()

    with torch.no_grad():
        feats = bb(x)  # x1,x2,x3,x4
    names = ["x1", "x2", "x3", "x4"]
    for nm, f in zip(names, feats):
        fm = f.mean(1)[0].cpu().numpy()  # channel-mean feature map
        save_heat(fm, os.path.join(OUT, f"feat_{nm}.png"), cmap="viridis")
        print(f"[feat] {nm}: {tuple(f.shape)} -> feat_{nm}.png")

    # --- 真实退化 proxy(网络内部,x3 语义层,与方法一致)
    for layer_name, f in [("x3", feats[2]), ("x4", feats[3])]:
        prox = degradation_proxies(f)[0].cpu().numpy()  # [6,h,w]
        for i, en in enumerate(EXPERT_NAMES):
            up = cv2.resize(prox[i], (S, S), interpolation=cv2.INTER_CUBIC)
            save_heat(up, os.path.join(OUT, f"proxy_{layer_name}_{en}.png"), cmap="inferno")
        print(f"[proxy] {layer_name}: 6 maps")

    # --- proxy 也在输入图层面算一份(最直观,可解释)
    g = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None]
    prox_img = degradation_proxies(g)[0].numpy()
    for i, en in enumerate(EXPERT_NAMES):
        save_heat(prox_img[i], os.path.join(OUT, f"proxyimg_{en}.png"), cmap="inferno")
    print("[proxy] input-level: 6 maps")
    print("ASSETS DONE ->", OUT)


if __name__ == "__main__":
    main()
