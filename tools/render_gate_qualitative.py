"""Selective-prediction qualitative figure for the reliability-gate paper.

Loads the physics-anchored engine's BUSI source checkpoint and runs REAL zero-shot
inference on the BUS-BRA target validation set (cross-domain). For every case it
computes the paper's label-free confidence signals (mean binary entropy,
confidence margin, ambiguous-band fraction, predicted foreground fraction) and the
IoU. Cases are ranked by the confidence-only combination the paper actually uses:
a 5-fold cross-validated logistic detector of transfer failure (IoU<0.5), scored
out-of-fold so no case sees its own label. The lowest-risk cases (gate KEEPS ->
trusted) are shown against the highest-risk cases (gate DEFERS -> flagged). Using
the multi-signal combination (rather than entropy alone) lets the foreground-
fraction signal catch confident-but-degenerate predictions that entropy misses.
No training of the segmenter, no fabricated masks: predictions come straight from
the frozen checkpoint.

Output: result/fig_qualitative_gate.png
"""
import os
import sys
import json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '1'

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import build_model
from dataloader.dataloader import getDataloader

SRC_EXP = "output/USEANet/busi/disc_mult02_busi"   # BUSI source model
TARGET = "BUSBRA"                                   # cross-domain target
NCOL = 4                                            # cases per row
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Args:
    pass


def load_model_and_loader():
    cfg = json.load(open(os.path.join(REPO, SRC_EXP, "config.json")))
    a = Args()
    for k, v in cfg.items():
        setattr(a, k, v)
    # cross-domain: keep the BUSI-trained weights, swap the loader to the target
    a.base_dir = f"hf_data/data/{TARGET}"
    a.dataset_name = TARGET
    a.val_file_dir = "val.txt"
    a.batch_size = 1
    pk = {'pretrained_model_path': a.pretrained_model_path} \
        if getattr(a, 'pretrained_model_path', None) else {}
    m = build_model(a, input_channel=a.input_channel, num_classes=a.num_classes, **pk).to(device)
    ck = torch.load(os.path.join(REPO, SRC_EXP, "checkpoint_best.pth"),
                    map_location=device, weights_only=False)
    m.load_state_dict(ck['state_dict'] if 'state_dict' in ck else ck)
    m.eval()
    _, val = getDataloader(a)
    return m, a, val


def label_free_signals(prob):
    """Paper's confidence-only signals, all computed from the prediction alone."""
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    ent = float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))
    margin = float(np.mean(np.abs(prob - 0.5)))                     # high = confident
    band = float(np.mean((prob > 0.3) & (prob < 0.7)))              # ambiguous fraction
    fgfrac = float(np.mean(prob > 0.5))                             # predicted fg fraction
    return ent, margin, band, fgfrac


def iou(pred_bin, gt_bin, eps=1e-6):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return float((inter + eps) / (union + eps))


def main():
    m, a, val = load_model_and_loader()
    rows = []
    with torch.no_grad():
        for s in val:
            case = s['case'][0] if isinstance(s['case'], (list, tuple)) else str(s['case'])
            inp, tgt = s['image'].to(device), s['label']
            out = m(inp)
            out = out[-1] if a.do_deeps else out
            prob = torch.sigmoid(out)[0, 0].cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8)
            gt = (tgt[0].squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            img = inp[0].mean(0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ent, margin, band_f, fgfrac = label_free_signals(prob)
            rows.append(dict(case=case, ent=ent, margin=margin, band=band_f,
                             fgfrac=fgfrac, iou=iou(pred, gt), img=img, gt=gt, pred=pred))
    print(f"scored {len(rows)} target cases")

    # Confidence-only combination the paper uses: a 5-fold CV logistic detector of
    # transfer failure (IoU<0.5), scored OUT-OF-FOLD so no case sees its own label.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    X = np.array([[r['ent'], r['margin'], r['band'], r['fgfrac']] for r in rows])
    y = np.array([1 if r['iou'] < 0.5 else 0 for r in rows])
    risk = np.zeros(len(rows))
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        risk[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    for r, rk in zip(rows, risk):
        r['risk'] = float(rk)
    print(f"combo AUROC (out-of-fold) = {roc_auc_score(y, risk):.3f}; failures = {int(y.sum())}/{len(y)}")

    rows.sort(key=lambda r: r['risk'])
    thr80 = float(np.quantile(risk, 0.80))       # gate defers the highest-risk 20%
    keep = rows[:NCOL]                            # lowest risk -> gate keeps
    defer = rows[-NCOL:][::-1]                    # highest risk -> gate defers

    fig, axes = plt.subplots(2, NCOL, figsize=(2.15 * NCOL, 4.7))
    band = {"keep": ("#2c7fb8", "KEEP\n(low risk)"),
            "defer": ("#c1272d", "DEFER\n(high risk)")}
    for ri, (grp, items) in enumerate([("keep", keep), ("defer", defer)]):
        col, label = band[grp]
        for ci in range(NCOL):
            ax = axes[ri][ci]
            r = items[ci]
            ax.imshow(r['img'], cmap='gray')
            ov = np.zeros((*r['pred'].shape, 4))
            ov[r['pred'] > 0.5] = (0.75, 0.22, 0.17, 0.45)
            ax.imshow(ov)
            ax.contour(r['gt'], levels=[0.5], colors=['#2ecc40'], linewidths=1.1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"risk={r['risk']:.2f}   IoU={r['iou']:.2f}",
                         fontsize=11, color=col)
        axes[ri][0].set_ylabel(label, fontsize=12, color=col, fontweight='bold',
                               labelpad=22, rotation=90, va='center')
    fig.suptitle(
        f"Selective prediction under cross-domain shift "
        f"(source-trained, zero-shot on {TARGET})\n"
        f"KEEP = 4 lowest-risk cases,  DEFER = 4 highest-risk "
        f"(gate defers the top-20% risk).  Green = ground truth,  red = prediction.",
        fontsize=12.5, y=1.03)
    plt.subplots_adjust(left=0.10, right=0.995, top=0.82, bottom=0.02,
                        wspace=0.05, hspace=0.32)
    out = os.path.join(REPO, "result/fig_qualitative_gate.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    # vector PDF: text + contours become vector; the B-mode images stay raster
    plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print("wrote", out, "and .pdf")
    print("KEEP  cases:", [(r['case'], round(r['risk'], 2), round(r['iou'], 2)) for r in keep])
    print("DEFER cases:", [(r['case'], round(r['risk'], 2), round(r['iou'], 2)) for r in defer])


if __name__ == '__main__':
    main()
