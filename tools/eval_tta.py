"""Inference-time accuracy boosters on an existing USEANet checkpoint (no retrain).

Evaluates the same BUSI val set + same metric methodology as main.py
(per-batch medpy-style IoU/Dice, size-weighted average via AverageMeter),
under three modes so the deltas are directly comparable:
  plain      : single forward, threshold 0.5
  tta        : average sigmoid over {orig, hflip, vflip, hvflip}
  tta_cc     : tta + keep largest connected component per image

Scores outputs[-1] (the refined primary head), matching main.py.
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from types import SimpleNamespace
from scipy import ndimage
from models import build_model
from dataloader.dataloader import getDataloader
from utils.util import AverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def batch_metrics(pred_bin, target_bin):
    """medpy-style global-over-batch IoU + Dice on binary uint8 arrays."""
    inter = float((pred_bin * target_bin).sum())
    psum, tsum = float(pred_bin.sum()), float(target_bin.sum())
    union = psum + tsum - inter
    iou = inter / union if union > 0 else 0.0
    dice = 2 * inter / (psum + tsum) if (psum + tsum) > 0 else 0.0
    return iou, dice


def largest_cc(mask_2d):
    lbl, n = ndimage.label(mask_2d)
    if n <= 1:
        return mask_2d
    sizes = ndimage.sum(np.ones_like(mask_2d), lbl, index=range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return (lbl == keep).astype(np.uint8)


def run(model, valloader, mode):
    iou_m, dice_m = AverageMeter(), AverageMeter()
    flips = [(False, False), (True, False), (False, True), (True, True)] if mode != "plain" else [(False, False)]
    with torch.no_grad():
        for sb in valloader:
            x = sb["image"].to(device); y = sb["label"].to(device)
            prob = torch.zeros_like(y)
            for hf, vf in flips:
                xi = x
                if hf: xi = torch.flip(xi, dims=[3])
                if vf: xi = torch.flip(xi, dims=[2])
                o = model(xi)
                o = o[-1] if isinstance(o, (list, tuple)) else o
                s = torch.sigmoid(o)
                if hf: s = torch.flip(s, dims=[3])
                if vf: s = torch.flip(s, dims=[2])
                prob = prob + s
            prob = prob / len(flips)
            pred = (prob > 0.5).cpu().numpy().astype(np.uint8)
            if mode == "tta_cc":
                for b in range(pred.shape[0]):
                    pred[b, 0] = largest_cc(pred[b, 0])
            tgt = (y > 0.5).cpu().numpy().astype(np.uint8)
            iou, dice = batch_metrics(pred, tgt)
            iou_m.update(iou, x.size(0)); dice_m.update(dice, x.size(0))
    return iou_m.avg, dice_m.avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    cfg = SimpleNamespace(
        model="USEANet", base_dir="hf_data/data/busi", dataset_name="busi",
        train_file_dir="train.txt", val_file_dir="val.txt", batch_size=8,
        img_size=256, num_classes=1, input_channel=3,
        pretrained_model_path="/home/chouheiwa/experiment/pretrain_model",
        model_id=115, do_deeps=True, seed=41,
    )
    model = build_model(config=cfg, input_channel=3, num_classes=1,
                        pretrained_model_path=cfg.pretrained_model_path).to(device)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    model.eval()
    _, valloader = getDataloader(cfg)
    print(f"ckpt {a.ckpt} (epoch={ck.get('epoch','?')})")
    for mode in ["plain", "tta", "tta_cc"]:
        iou, dice = run(model, valloader, mode)
        print(f"{mode:8s}  IoU={iou:.4f}  Dice={dice:.4f}")


if __name__ == "__main__":
    main()
