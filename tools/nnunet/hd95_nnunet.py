#!/usr/bin/env python
"""Offline HD95 for nnU-Net val predictions, computed in the 256-px space.

nnU-Net predicts at each image's ORIGINAL resolution (variable per case), while
every U-Bench baseline validated at 256x256. HD95 is an absolute pixel distance
and is resolution-sensitive, so to be comparable we resize BOTH the prediction
and the GT to 256x256 (nearest-neighbour, mask-preserving) before measuring —
the methodology decision recorded in memory `hd95-backfill` (2026-06-22).

Same empty-prediction convention as tools/offline_hd95.py: GT is non-empty
(ultrasound lesions), an empty prediction is PENALIZED with the 256-px diagonal
(~362) rather than skipped. IoU is also recomputed at 256 as a sanity check
against result_nnunet.csv (should match closely; tiny drift from the resize).

Appends one row per (dataset, seed) to result/result_hd95.csv with
modelname=nnUNet, matching the baseline/PUMA schema so the final table merges.
"""
import argparse
import csv
import json
import math
import os

import cv2
import numpy as np
from medpy.metric.binary import hd95 as medpy_hd95

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_ROOT = os.environ.get("nnUNet_raw", os.path.join(REPO, "nnunet", "raw"))
RESULT_ROOT = os.environ.get("nnUNet_results", os.path.join(REPO, "nnunet", "results"))
RESULT_CSV = os.path.join(REPO, "result", "result_hd95.csv")
SIZE = 256

DATASET_NAMES = {
    "busi": "Dataset501_busi",
    "bus": "Dataset502_bus",
    "BUSBRA": "Dataset503_BUSBRA",
    "tuscui": "Dataset504_tuscui",
}


def to256(mask):
    """Binarize then nearest-neighbour resize to 256x256 (mask-preserving)."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.shape[:2] != (SIZE, SIZE):
        m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def hd95_case(pred, gt):
    p, g = pred.sum(), gt.sum()
    if p == 0 and g == 0:
        return 0.0
    if p == 0 or g == 0:
        return float(math.sqrt(SIZE * SIZE + SIZE * SIZE))  # diagonal penalty
    return float(medpy_hd95(pred, gt))


def iou_case(pred, gt):
    inter = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - inter
    return float(inter) / float(union) if union > 0 else 0.0


def evaluate(ds, seed):
    name = DATASET_NAMES[ds]
    raw = os.path.join(RAW_ROOT, name)
    pred_dir = os.path.join(RESULT_ROOT, name, f"pred_s{seed}")
    labels_dir = os.path.join(raw, "labelsTr")
    with open(os.path.join(raw, "case_map.json")) as f:
        case_map = json.load(f)
    hds, ious, n_empty = [], [], 0
    for clean, info in case_map.items():
        if not info["val"]:
            continue
        pred = cv2.imread(os.path.join(pred_dir, f"{clean}.png"), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(labels_dir, f"{clean}.png"), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            raise FileNotFoundError(f"missing prediction {pred_dir}/{clean}.png")
        if gt is None:
            raise FileNotFoundError(f"missing GT {labels_dir}/{clean}.png")
        p, g = to256(pred), to256(gt)
        if p.sum() == 0 and g.sum() > 0:
            n_empty += 1
        hds.append(hd95_case(p, g))
        ious.append(iou_case(p, g))
    if not hds:
        raise RuntimeError(f"{ds}: no val cases in case_map")
    return float(np.mean(hds)), float(np.mean(ious)), len(hds), n_empty


def append_row(ds, seed, hd, iou_check, n, n_empty):
    os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
    new = not os.path.exists(RESULT_CSV)
    with open(RESULT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["modelname", "dataset", "seed", "exp_name",
                        "hd95", "iou_check", "n_val", "n_empty_pred"])
        w.writerow(["nnUNet", ds, seed, f"pred_s{seed}",
                    f"{hd:.6f}", f"{iou_check:.6f}", n, n_empty])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_NAMES))
    ap.add_argument("--seed", required=True, type=int)
    a = ap.parse_args()
    hd, iou_check, n, n_empty = evaluate(a.dataset, a.seed)
    append_row(a.dataset, a.seed, hd, iou_check, n, n_empty)
    print(f"[hd95-nnunet] {a.dataset} s{a.seed} HD95={hd:.4f} "
          f"IoU_check@256={iou_check:.4f} n={n} empty={n_empty} -> {RESULT_CSV}")


if __name__ == "__main__":
    main()
