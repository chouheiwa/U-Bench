#!/usr/bin/env python
"""Offline HD95 backfill from a trained cell's checkpoint_best.pth.

The matrix's main.py never recorded a boundary metric — metrics_medpy.get_metrics
returns only IoU/Dice/SE/PC/F1/SP/ACC (it imports medpy `hd` but never calls it).
HD95 needs no retraining: load the best checkpoint, run val inference at the same
resolution the cell validated on (val_transform = Resize+Normalize, identical to
training), and compute medpy hd95 per case.

Empty-prediction convention (user decision 2026-06-22): these ultrasound datasets
always have a non-empty GT lesion, so only the prediction can be empty. An empty
prediction is PENALIZED with the image diagonal (sqrt(H^2+W^2)) at that cell's
resolution — it is not skipped (skipping would reward high-recall-failure methods).

We also recompute IoU as a sanity check that the right checkpoint loaded and the
val pipeline matches (should land within a small tolerance of the recorded best_iou).

Output: one row appended to result/result_hd95.csv:
  modelname,dataset,seed,exp_name,hd95,iou_check,n_val,n_empty_pred
"""
import argparse
import csv
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from medpy.metric.binary import hd95 as medpy_hd95

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # so `from models import ...` resolves like main.py
RESULT_CSV = os.path.join(REPO, "result", "result_hd95.csv")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hd95_case(pred, gt):
    """Per-case HD95 with the image-diagonal penalty for empty predictions.

    pred, gt: 2D uint8 arrays in {0,1}. GT is assumed non-empty (ultrasound
    lesion datasets); both-empty -> 0; pred-empty (GT present) -> diagonal.
    """
    pred = (np.asarray(pred) > 0).astype(np.uint8)
    gt = (np.asarray(gt) > 0).astype(np.uint8)
    p, g = pred.sum(), gt.sum()
    if p == 0 and g == 0:
        return 0.0
    if p == 0 or g == 0:
        h, w = gt.shape[-2:]
        return float(math.sqrt(h * h + w * w))  # diagonal penalty
    return float(medpy_hd95(pred, gt))


def iou_case(pred, gt):
    pred = (np.asarray(pred) > 0).astype(np.uint8)
    gt = (np.asarray(gt) > 0).astype(np.uint8)
    inter = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - inter
    return float(inter) / float(union) if union > 0 else 0.0


def build_args(a):
    """Namespace with every field build_model / getDataloader / load read."""
    return SimpleNamespace(
        model=a.model, model_id=a.model_id, img_size=a.img_size,
        base_dir=a.base_dir, dataset_name=a.dataset_name,
        batch_size=a.batch_size, seed=a.seed,
        input_channel=3, num_classes=1, do_deeps=False,
        pretrained_model_path=None, exp_save_dir=a.exp_save_dir,
        train_file_dir="train.txt", val_file_dir="val.txt",
    )


def evaluate_cell(a):
    from models import build_model
    from dataloader.dataloader import getDataloader

    args = build_args(a)
    _pre = {}
    model = build_model(args, input_channel=args.input_channel,
                        num_classes=args.num_classes, **_pre).to(device)
    ckpt = torch.load(os.path.join(args.exp_save_dir, "checkpoint_best.pth"),
                      map_location=device, weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    _, valloader = getDataloader(args)
    hds, ious, n_empty = [], [], 0
    with torch.no_grad():
        for batch in valloader:
            inp, target = batch["image"].to(device), batch["label"]
            out = model(inp)
            # Deep-supervision models (e.g. USEANet/PUMA-Net) return a list of
            # outputs; the final one is the full-resolution prediction.
            if isinstance(out, (list, tuple)):
                out = out[-1]
            prob = torch.sigmoid(out).cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8)          # (B,1,H,W)
            gt = (target.numpy() > 0.5).astype(np.uint8)  # (B,1,H,W)
            for b in range(pred.shape[0]):
                p2 = pred[b, 0]
                g2 = gt[b, 0]
                if p2.sum() == 0 and g2.sum() > 0:
                    n_empty += 1
                hds.append(hd95_case(p2, g2))
                ious.append(iou_case(p2, g2))
    return float(np.mean(hds)), float(np.mean(ious)), len(hds), n_empty


def append_row(a, hd, iou_check, n, n_empty):
    os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
    new = not os.path.exists(RESULT_CSV)
    with open(RESULT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["modelname", "dataset", "seed", "exp_name",
                        "hd95", "iou_check", "n_val", "n_empty_pred"])
        w.writerow([a.model, a.dataset_name, a.seed, a.exp_name,
                    f"{hd:.6f}", f"{iou_check:.6f}", n, n_empty])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_id", type=int, required=True)
    ap.add_argument("--img_size", type=int, required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--exp_save_dir", required=True)
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    a = ap.parse_args()
    hd, iou_check, n, n_empty = evaluate_cell(a)
    append_row(a, hd, iou_check, n, n_empty)
    print(f"[hd95] {a.model} {a.dataset_name} s{a.seed} HD95={hd:.4f} "
          f"IoU_check={iou_check:.4f} n={n} empty={n_empty} -> {RESULT_CSV}")


if __name__ == "__main__":
    main()
