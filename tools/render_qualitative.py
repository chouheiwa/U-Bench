"""Qualitative BUSI segmentation comparison: PUMA-Net vs H2Former vs U-Net.
Loads each model's best checkpoint, predicts on a few val cases, and renders a
grid with GT (green) and prediction (red) contours over the ultrasound image.
Zero training; reuses the frozen best checkpoints."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '1'

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import build_model
from dataloader.dataloader import getDataloader

MODELS = [
    ("USEANet", "disc_mult02_busi", "PUMA-Net (ours)"),
    ("H2Former", "baseline_s41", "H2Former"),
    ("MSLAU_Net", "mtx_s41", "MSLAU-Net"),
    ("CMUNeXt", "baseline_s41", "CMUNeXt"),
    ("U_Net", "baseline_s41", "U-Net"),
]
CASES = ["malignant (185)", "malignant (130)", "benign (339)"]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Args:
    pass


def load(model_name, exp):
    d = f"./output/{model_name}/busi/{exp}"
    cfg = json.load(open(os.path.join(d, "config.json")))
    a = Args()
    for k, v in cfg.items():
        setattr(a, k, v)
    pk = {'pretrained_model_path': a.pretrained_model_path} \
        if getattr(a, 'pretrained_model_path', None) else {}
    m = build_model(a, input_channel=a.input_channel, num_classes=a.num_classes, **pk).to(device)
    ck = torch.load(os.path.join(d, "checkpoint_best.pth"), map_location=device, weights_only=False)
    m.load_state_dict(ck['state_dict'] if 'state_dict' in ck else ck)
    m.eval()
    return m, a


def collect(model_name, exp):
    """Return {case: pred_mask} plus shared {case: (img, gt)} for target cases."""
    m, a = load(model_name, exp)
    _, val = getDataloader(a)
    preds, shared = {}, {}
    with torch.no_grad():
        for s in val:
            case = s['case'][0] if isinstance(s['case'], (list, tuple)) else str(s['case'])
            if case not in CASES:
                continue
            inp, tgt = s['image'].to(device), s['label']
            out = m(inp)
            out = out[-1] if a.do_deeps else out
            pred = (torch.sigmoid(out)[0, 0].cpu().numpy() > 0.5).astype(float)
            preds[case] = pred
            img = inp[0].mean(0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            gt = tgt[0].squeeze().cpu().numpy()
            shared[case] = (img, (gt > 0.5).astype(float))
            if len(preds) == len(CASES):
                break
    return preds, shared


def main():
    all_preds, shared = {}, {}
    for mn, exp, _ in MODELS:
        p, sh = collect(mn, exp)
        all_preds[mn] = p
        shared.update(sh)

    ncol = 1 + len(MODELS)
    nrow = len(CASES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol, 1.9 * nrow))
    col_titles = ["US image + GT"] + [lab for _, _, lab in MODELS]
    for r, case in enumerate(CASES):
        img, gt = shared[case]
        for c in range(ncol):
            ax = axes[r][c]
            ax.imshow(img, cmap='gray')
            # prediction as a semi-transparent red region (like the arch figure)
            if c >= 1:
                pred = all_preds[MODELS[c - 1][0]].get(case)
                if pred is not None:
                    ov = np.zeros((*pred.shape, 4))
                    ov[pred > 0.5] = (0.75, 0.22, 0.17, 0.45)
                    ax.imshow(ov)
            # ground truth as a green contour on top
            ax.contour(gt, levels=[0.5], colors=['#2ecc40'], linewidths=1.2)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8,
                             fontweight='bold' if c == 1 else 'normal',
                             color='#c0392b' if c == 1 else 'black')
            ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.5, 0.005, r"green = ground-truth contour   |   red = prediction region",
             ha='center', fontsize=8)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    out = "paper/puma-net/figs/qualitative.pdf"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print("wrote", out, "cases:", CASES)


if __name__ == '__main__':
    main()
