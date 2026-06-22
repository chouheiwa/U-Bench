#!/usr/bin/env python
"""Evaluate nnU-Net val predictions with U-Bench's exact IoU and append a CSV row.

IoU is bit-for-bit U-Bench's formula (utils/metrics_medpy.py:19-21): binary
masks, intersection/union, union==0 -> 0, per-case then mean over the val set.
nnU-Net predictions are argmax labels (0/1), so no sigmoid/threshold is needed.

Results go to a SEPARATE result/result_nnunet.csv (modelname,dataset,seed,
best_iou) to avoid the shared 33/34-col CSV column-alignment fragility; merged
into the final table at report time.
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_ROOT = os.environ.get("nnUNet_raw", os.path.join(REPO, "nnunet", "raw"))
RESULT_CSV = os.path.join(REPO, "result", "result_nnunet.csv")

DATASET_NAMES = {
    "busi": "Dataset501_busi",
    "bus": "Dataset502_bus",
    "BUSBRA": "Dataset503_BUSBRA",
    "tuscui": "Dataset504_tuscui",
}


def iou_score(output, target):
    """U-Bench IoU (utils/metrics_medpy.py:19-21). Inputs are binary {0,1}."""
    output = (np.asarray(output) > 0).astype(np.uint8)
    target = (np.asarray(target) > 0).astype(np.uint8)
    intersection = np.sum(output * target)
    union = np.sum(output) + np.sum(target) - intersection
    return float(intersection) / float(union) if union > 0 else 0.0


def evaluate(ds, pred_dir):
    """Mean IoU over the val cases of dataset ds, comparing pred_dir vs raw GT."""
    raw = os.path.join(RAW_ROOT, DATASET_NAMES[ds])
    with open(os.path.join(raw, "case_map.json")) as f:
        case_map = json.load(f)
    labels_dir = os.path.join(raw, "labelsTr")
    ious = []
    for clean, info in case_map.items():
        if not info["val"]:
            continue
        pred_p = os.path.join(pred_dir, f"{clean}.png")
        gt_p = os.path.join(labels_dir, f"{clean}.png")
        pred = cv2.imread(pred_p, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_p, cv2.IMREAD_GRAYSCALE)
        if pred is None:
            raise FileNotFoundError(f"missing prediction {pred_p}")
        if gt is None:
            raise FileNotFoundError(f"missing GT {gt_p}")
        ious.append(iou_score(pred, gt))
    if not ious:
        raise RuntimeError(f"{ds}: no val cases found in case_map")
    return float(np.mean(ious)), len(ious)


def append_row(ds, seed, iou):
    os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
    new = not os.path.exists(RESULT_CSV)
    with open(RESULT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["modelname", "dataset", "seed", "best_iou"])
        w.writerow(["nnUNet", ds, seed, f"{iou:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_NAMES))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--pred-dir", required=True)
    args = ap.parse_args()
    iou, n = evaluate(args.dataset, args.pred_dir)
    append_row(args.dataset, args.seed, iou)
    print(f"[eval] {args.dataset} seed={args.seed} val_n={n} IoU={iou:.6f} "
          f"-> {RESULT_CSV}")


if __name__ == "__main__":
    main()
