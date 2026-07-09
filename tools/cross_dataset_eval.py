#!/usr/bin/env python
"""Naive cross-dataset (zero-shot) transfer evaluation for the new-paper P0 lower bound.

Loads a trained checkpoint (source dataset) and evaluates it, WITHOUT any adaptation,
on a target dataset's val split. This is the "未自适应跨域下界" — the baseline the
physics-calibrated routing + source-free TTA (paper contributions 2/3) must beat.

Non-destructive: reads checkpoint_best.pth from the source cell's exp dir but never
writes there (unlike main.py --just_for_test, which would overwrite config.json via
init_dir). Reuses getZeroShotDataloader so BUSBRA-style datasets dispatch correctly.

Metrics per case: IoU, Dice, Recall (sensitivity), HD95 (image-diagonal penalty for
empty predictions, matching tools/offline_hd95.py's convention). One row appended to
result/result_cross_dataset.csv:
  modelname,source,target,seed,exp_name,iou,dice,recall,hd95,n_val,n_empty_pred
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
sys.path.insert(0, REPO)
RESULT_CSV = os.path.join(REPO, "result", "result_cross_dataset.csv")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hd95_case(pred, gt):
    pred = (np.asarray(pred) > 0).astype(np.uint8)
    gt = (np.asarray(gt) > 0).astype(np.uint8)
    p, g = pred.sum(), gt.sum()
    if p == 0 and g == 0:
        return 0.0
    if p == 0 or g == 0:
        h, w = gt.shape[-2:]
        return float(math.sqrt(h * h + w * w))
    return float(medpy_hd95(pred, gt))


def iou_dice_recall(pred, gt):
    pred = (np.asarray(pred) > 0).astype(np.uint8)
    gt = (np.asarray(gt) > 0).astype(np.uint8)
    inter = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - inter
    iou = float(inter) / float(union) if union > 0 else 0.0
    denom = np.sum(pred) + np.sum(gt)
    dice = 2.0 * float(inter) / float(denom) if denom > 0 else 0.0
    recall = float(inter) / float(np.sum(gt)) if np.sum(gt) > 0 else 0.0
    return iou, dice, recall


def build_args(a):
    return SimpleNamespace(
        model=a.model, model_id=a.model_id, img_size=a.img_size,
        base_dir=a.base_dir, dataset_name=a.source,
        batch_size=1, seed=a.seed,
        input_channel=3, num_classes=1, do_deeps=False,
        pretrained_model_path=a.pretrained_model_path, exp_save_dir=a.exp_save_dir,
        train_file_dir="train.txt", val_file_dir=a.val_file_dir,
        # zero-shot loader fields (target):
        zero_shot_dataset_name=a.target, zero_shot_base_dir=a.target_base_dir,
    )


def evaluate(a):
    from models import build_model
    from dataloader.dataloader import getZeroShotDataloader

    args = build_args(a)
    _pre = {'pretrained_model_path': args.pretrained_model_path} if args.pretrained_model_path else {}
    model = build_model(args, input_channel=args.input_channel,
                        num_classes=args.num_classes, **_pre).to(device)
    ckpt = torch.load(os.path.join(args.exp_save_dir, "checkpoint_best.pth"),
                      map_location=device, weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    valloader = getZeroShotDataloader(args)
    ious, dices, recalls, hds, n_empty = [], [], [], [], 0
    with torch.no_grad():
        for batch in valloader:
            inp, target = batch["image"].to(device), batch["label"]
            out = model(inp)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            prob = torch.sigmoid(out).cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8)
            gt = (target.numpy() > 0.5).astype(np.uint8)
            for b in range(pred.shape[0]):
                p2, g2 = pred[b, 0], gt[b, 0]
                if p2.sum() == 0 and g2.sum() > 0:
                    n_empty += 1
                iou, dice, recall = iou_dice_recall(p2, g2)
                ious.append(iou); dices.append(dice); recalls.append(recall)
                hds.append(hd95_case(p2, g2))
    # Per-case arrays returned alongside aggregates so callers can dump them for
    # paired significance tests (Wilcoxon). Case order is deterministic given the
    # fixed seed, so index i is the same target case across models -> pairable.
    return (float(np.mean(ious)), float(np.mean(dices)), float(np.mean(recalls)),
            float(np.mean(hds)), len(ious), n_empty, ious, dices, recalls, hds)


def append_row(a, iou, dice, recall, hd, n, n_empty):
    os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
    new = not os.path.exists(RESULT_CSV)
    with open(RESULT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["modelname", "source", "target", "seed", "exp_name",
                        "iou", "dice", "recall", "hd95", "n_val", "n_empty_pred"])
        w.writerow([a.model, a.source, a.target, a.seed, a.exp_name,
                    f"{iou:.6f}", f"{dice:.6f}", f"{recall:.6f}", f"{hd:.6f}", n, n_empty])


def append_percase(a, csv_path, ious, dices, recalls, hds):
    """One row per target case -> paired Wilcoxon fodder. Additive; never touches
    the aggregate CSV. Keyed by (exp_name, source, target, seed, case_idx) so a
    later join can pair the same case across two models."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["exp_name", "source", "target", "seed", "case_idx",
                        "iou", "dice", "recall", "hd95"])
        for i in range(len(ious)):
            w.writerow([a.exp_name, a.source, a.target, a.seed, i,
                        f"{ious[i]:.6f}", f"{dices[i]:.6f}",
                        f"{recalls[i]:.6f}", f"{hds[i]:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="USEANet")
    ap.add_argument("--model_id", type=int, default=115)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--source", required=True)
    ap.add_argument("--base_dir", required=True, help="source dataset dir (unused for eval, kept for build_args)")
    ap.add_argument("--target", required=True)
    ap.add_argument("--target_base_dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--exp_save_dir", required=True)
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--val_file_dir", default="val.txt")
    ap.add_argument("--pretrained_model_path", default="./pretrained")
    ap.add_argument("--dump_cases", action="store_true",
                    help="also write per-case IoU/Dice/Recall/HD95 for paired significance tests")
    ap.add_argument("--percase_csv", default=os.path.join(REPO, "result", "percase_cross_dataset.csv"))
    a = ap.parse_args()
    iou, dice, recall, hd, n, n_empty, ci, cd, cr, ch = evaluate(a)
    append_row(a, iou, dice, recall, hd, n, n_empty)
    if a.dump_cases:
        append_percase(a, a.percase_csv, ci, cd, cr, ch)
    print(f"[xds] {a.model} {a.source}->{a.target} s{a.seed} "
          f"IoU={iou:.4f} Dice={dice:.4f} Recall={recall:.4f} HD95={hd:.4f} "
          f"n={n} empty={n_empty}{' +percase' if a.dump_cases else ''} -> {RESULT_CSV}")


if __name__ == "__main__":
    main()
